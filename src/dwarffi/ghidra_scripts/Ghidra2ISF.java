// Exports the current Ghidra program's data types, symbols, and functions to
// Volatility-style Intermediate Symbol File (ISF) JSON.
//
// Usage from Ghidra Script Manager:
//   Run the script and choose an output .json file when prompted.
//
// Usage from analyzeHeadless:
//   analyzeHeadless /tmp/proj Proj -import sample.bin \
//     -scriptPath /path/to/src/dwarffi/ghidra_scripts \
//     -postScript Ghidra2ISF.java /tmp/sample.isf.json
//
// Optional arguments:
//   --types-only    Export only type sections
//   --no-symbols    Do not export symbols
//   --no-functions  Do not export functions

import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.data.Array;
import ghidra.program.model.data.BitFieldDataType;
import ghidra.program.model.data.BooleanDataType;
import ghidra.program.model.data.CharDataType;
import ghidra.program.model.data.DataType;
import ghidra.program.model.data.DataTypeComponent;
import ghidra.program.model.data.DataTypeManager;
import ghidra.program.model.data.DefaultDataType;
import ghidra.program.model.data.Enum;
import ghidra.program.model.data.FloatDataType;
import ghidra.program.model.data.FunctionDefinition;
import ghidra.program.model.data.ParameterDefinition;
import ghidra.program.model.data.Pointer;
import ghidra.program.model.data.Structure;
import ghidra.program.model.data.TypeDef;
import ghidra.program.model.data.Union;
import ghidra.program.model.data.VoidDataType;
import ghidra.program.model.listing.Data;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.listing.Listing;
import ghidra.program.model.listing.Parameter;
import ghidra.program.model.listing.Program;
import ghidra.program.model.symbol.Symbol;
import ghidra.program.model.symbol.SymbolIterator;
import ghidra.program.model.symbol.SymbolTable;
import ghidra.program.model.symbol.SymbolType;
import ghidra.util.task.TaskMonitor;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;

public class Ghidra2ISF extends GhidraScript {
    private static final String TOOL_NAME = "ghidra2isf";
    private static final String TOOL_VERSION = "0.1.0";
    private static final String FORMAT_VERSION = "6.2.0";

    @Override
    protected void run() throws Exception {
        if (currentProgram == null) {
            printerr("No current program is open.");
            return;
        }

        ExportOptions options = ExportOptions.fromArgs(getScriptArgs());
        if (options.outputFile == null) {
            options.outputFile = askFile("Save ISF JSON", "Save");
        }

        IsfExporter exporter = new IsfExporter(currentProgram, monitor);
        Map<String, Object> document = exporter.export(options);
        writeJson(options.outputFile, document);

        println(String.format(
            "Wrote ISF to %s (%d base types, %d user types, %d enums, %d typedefs, %d symbols, %d functions)",
            options.outputFile.getAbsolutePath(),
            exporter.baseTypes.size(),
            exporter.userTypes.size(),
            exporter.enums.size(),
            exporter.typedefs.size(),
            exporter.symbols.size(),
            exporter.functions.size()
        ));
    }

    private static void writeJson(File outputFile, Map<String, Object> document) throws IOException {
        File parent = outputFile.getAbsoluteFile().getParentFile();
        if (parent != null) {
            Files.createDirectories(parent.toPath());
        }

        try (BufferedWriter writer = Files.newBufferedWriter(outputFile.toPath(), StandardCharsets.UTF_8)) {
            JsonWriter.write(document, writer);
            writer.write('\n');
        }
    }

    private static final class ExportOptions {
        File outputFile;
        boolean exportSymbols = true;
        boolean exportFunctions = true;

        static ExportOptions fromArgs(String[] args) {
            ExportOptions options = new ExportOptions();
            if (args == null) {
                return options;
            }

            for (String arg : args) {
                if ("--types-only".equals(arg)) {
                    options.exportSymbols = false;
                    options.exportFunctions = false;
                } else if ("--no-symbols".equals(arg)) {
                    options.exportSymbols = false;
                } else if ("--no-functions".equals(arg)) {
                    options.exportFunctions = false;
                } else if (options.outputFile == null) {
                    options.outputFile = new File(arg);
                } else {
                    throw new IllegalArgumentException("Unexpected argument: " + arg);
                }
            }
            return options;
        }
    }

    private static final class IsfExporter {
        private final Program program;
        private final TaskMonitor monitor;
        private final String endian;
        private final int pointerSize;

        final TreeMap<String, Object> baseTypes = new TreeMap<>();
        final TreeMap<String, Object> userTypes = new TreeMap<>();
        final TreeMap<String, Object> enums = new TreeMap<>();
        final TreeMap<String, Object> typedefs = new TreeMap<>();
        final TreeMap<String, Object> symbols = new TreeMap<>();
        final TreeMap<String, Object> functions = new TreeMap<>();

        IsfExporter(Program program, TaskMonitor monitor) {
            this.program = program;
            this.monitor = monitor;
            this.endian = program.getLanguage().isBigEndian() ? "big" : "little";
            this.pointerSize = Math.max(program.getDefaultPointerSize(), 1);
        }

        Map<String, Object> export(ExportOptions options) throws Exception {
            exportTypes();
            if (options.exportSymbols) {
                exportSymbols();
            }
            if (options.exportFunctions) {
                exportFunctions();
            }

            LinkedHashMap<String, Object> document = new LinkedHashMap<>();
            document.put("metadata", metadata());
            document.put("base_types", baseTypes);
            document.put("user_types", userTypes);
            document.put("enums", enums);
            document.put("symbols", symbols);
            document.put("functions", functions);
            document.put("typedefs", typedefs);
            return document;
        }

        private Map<String, Object> metadata() {
            LinkedHashMap<String, Object> producer = new LinkedHashMap<>();
            producer.put("name", TOOL_NAME);
            producer.put("version", TOOL_VERSION);
            producer.put("ghidra_version", System.getProperty("application.version", "unknown"));

            LinkedHashMap<String, Object> source = new LinkedHashMap<>();
            source.put("kind", "ghidra_program");
            source.put("name", program.getName());

            LinkedHashMap<String, Object> ghidra = new LinkedHashMap<>();
            ghidra.put("types", listOf(source));
            ghidra.put("symbols", listOf(source));

            LinkedHashMap<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("producer", producer);
            metadata.put("format", FORMAT_VERSION);
            metadata.put("ghidra", ghidra);
            return metadata;
        }

        private static List<Object> listOf(Object value) {
            ArrayList<Object> values = new ArrayList<>();
            values.add(value);
            return values;
        }

        private void exportTypes() throws Exception {
            ensureVoid();
            ensurePointer();

            DataTypeManager dataTypeManager = program.getDataTypeManager();
            Iterator<DataType> iterator = dataTypeManager.getAllDataTypes();
            while (iterator.hasNext()) {
                monitor.checkCancelled();
                DataType dataType = iterator.next();
                if (dataType == null || dataType instanceof DefaultDataType) {
                    continue;
                }
                exportDataType(dataType);
            }
        }

        private void exportDataType(DataType dataType) {
            if (dataType instanceof Structure) {
                exportComposite((Structure) dataType, "struct");
            } else if (dataType instanceof Union) {
                exportComposite((Union) dataType, "union");
            } else if (dataType instanceof Enum) {
                exportEnum((Enum) dataType);
            } else if (dataType instanceof TypeDef) {
                exportTypedef((TypeDef) dataType);
            } else if (isBaseLike(dataType)) {
                ensureBase(dataType);
            } else if (dataType instanceof Pointer) {
                ensurePointer();
                typeRef(((Pointer) dataType).getDataType());
            } else if (dataType instanceof Array) {
                typeRef(((Array) dataType).getDataType());
            }
        }

        private void exportComposite(DataType composite, String kind) {
            String name = typeName(composite);
            if (userTypes.containsKey(name)) {
                return;
            }

            TreeMap<String, Object> fields = new TreeMap<>();
            DataTypeComponent[] components;
            if (composite instanceof Structure) {
                components = ((Structure) composite).getComponents();
            } else {
                components = ((Union) composite).getComponents();
            }

            int anonymousCount = 0;
            for (DataTypeComponent component : components) {
                if (component == null) {
                    continue;
                }

                DataType fieldType = component.getDataType();
                if (fieldType == null || fieldType instanceof DefaultDataType) {
                    continue;
                }

                String fieldName = component.getFieldName();
                boolean anonymous = fieldName == null || fieldName.isEmpty();
                if (anonymous) {
                    fieldName = "unnamed_field_" + anonymousCount;
                    anonymousCount += 1;
                }

                LinkedHashMap<String, Object> field = new LinkedHashMap<>();
                field.put("type", fieldTypeRef(component));
                field.put("offset", Math.max(component.getOffset(), 0));
                if (anonymous) {
                    field.put("anonymous", true);
                }
                fields.put(fieldName, field);
            }

            LinkedHashMap<String, Object> record = new LinkedHashMap<>();
            record.put("size", Math.max(composite.getLength(), 0));
            record.put("fields", fields);
            record.put("kind", kind);
            userTypes.put(name, record);
        }

        private Map<String, Object> fieldTypeRef(DataTypeComponent component) {
            DataType dataType = component.getDataType();
            if (dataType instanceof BitFieldDataType) {
                BitFieldDataType bitField = (BitFieldDataType) dataType;
                LinkedHashMap<String, Object> wrapper = new LinkedHashMap<>();
                wrapper.put("kind", "bitfield");
                wrapper.put("bit_length", firstPositiveInt(bitField, 0, "getDeclaredBitSize", "getBitSize"));
                wrapper.put("bit_position", firstPositiveInt(component, 0, "getBitOffset"));
                wrapper.put("type", typeRef(bitField.getBaseDataType()));
                return wrapper;
            }
            return typeRef(dataType);
        }

        private void exportEnum(Enum enumType) {
            String name = typeName(enumType);
            if (enums.containsKey(name)) {
                return;
            }

            TreeMap<String, Object> constants = new TreeMap<>();
            for (String constantName : enumType.getNames()) {
                constants.put(constantName, enumType.getValue(constantName));
            }

            LinkedHashMap<String, Object> record = new LinkedHashMap<>();
            record.put("size", Math.max(enumType.getLength(), 0));
            record.put("base", enumBaseName(enumType));
            record.put("constants", constants);
            enums.put(name, record);
        }

        private String enumBaseName(Enum enumType) {
            int length = Math.max(enumType.getLength(), 0);
            boolean signed = false;
            for (String constantName : enumType.getNames()) {
                if (enumType.getValue(constantName) < 0) {
                    signed = true;
                    break;
                }
            }
            String name = (signed ? "int" : "uint") + (length * 8) + "_t";
            ensureSyntheticBase(name, length, "int", signed);
            return name;
        }

        private void exportTypedef(TypeDef typeDef) {
            String name = typeName(typeDef);
            if (typedefs.containsKey(name)) {
                return;
            }
            typedefs.put(name, typeRef(typeDef.getBaseDataType()));
        }

        private void exportSymbols() throws Exception {
            SymbolTable symbolTable = program.getSymbolTable();
            Listing listing = program.getListing();
            SymbolIterator iterator = symbolTable.getAllSymbols(true);
            while (iterator.hasNext()) {
                monitor.checkCancelled();
                Symbol symbol = iterator.next();
                if (symbol == null || symbol.isExternal() || symbol.getAddress() == null) {
                    continue;
                }
                if (symbol.getSymbolType() == SymbolType.FUNCTION) {
                    continue;
                }
                Address address = symbol.getAddress();
                if (!program.getMemory().contains(address)) {
                    continue;
                }

                LinkedHashMap<String, Object> record = new LinkedHashMap<>();
                Data data = listing.getDataAt(address);
                if (data != null && data.getDataType() != null) {
                    record.put("type", typeRef(data.getDataType()));
                }
                record.put("address", address.getOffset());
                symbols.put(symbol.getName(true), record);
            }
        }

        private void exportFunctions() throws Exception {
            FunctionIterator iterator = program.getListing().getFunctions(true);
            while (iterator.hasNext()) {
                monitor.checkCancelled();
                Function function = iterator.next();
                if (function == null || function.isExternal()) {
                    continue;
                }

                ArrayList<Object> parameters = new ArrayList<>();
                for (Parameter parameter : function.getParameters()) {
                    LinkedHashMap<String, Object> p = new LinkedHashMap<>();
                    p.put("name", parameter.getName());
                    p.put("type", typeRef(parameter.getDataType()));
                    parameters.add(p);
                }

                LinkedHashMap<String, Object> record = new LinkedHashMap<>();
                record.put("address", function.getEntryPoint().getOffset());
                record.put("return_type", typeRef(function.getReturnType()));
                record.put("parameters", parameters);
                functions.put(function.getName(true), record);
            }
        }

        private Map<String, Object> typeRef(DataType dataType) {
            LinkedHashMap<String, Object> result = new LinkedHashMap<>();
            if (dataType == null || dataType instanceof VoidDataType) {
                ensureVoid();
                result.put("kind", "base");
                result.put("name", "void");
                return result;
            }

            if (dataType instanceof TypeDef) {
                exportTypedef((TypeDef) dataType);
                result.put("kind", "typedef");
                result.put("name", typeName(dataType));
                return result;
            }

            if (dataType instanceof Pointer) {
                ensurePointer();
                result.put("kind", "pointer");
                result.put("subtype", typeRef(((Pointer) dataType).getDataType()));
                return result;
            }

            if (dataType instanceof Array) {
                Array array = (Array) dataType;
                result.put("kind", "array");
                result.put("count", Math.max(array.getNumElements(), 0));
                result.put("subtype", typeRef(array.getDataType()));
                return result;
            }

            if (dataType instanceof Structure) {
                exportComposite((Structure) dataType, "struct");
                result.put("kind", "struct");
                result.put("name", typeName(dataType));
                return result;
            }

            if (dataType instanceof Union) {
                exportComposite((Union) dataType, "union");
                result.put("kind", "union");
                result.put("name", typeName(dataType));
                return result;
            }

            if (dataType instanceof Enum) {
                exportEnum((Enum) dataType);
                result.put("kind", "enum");
                result.put("name", typeName(dataType));
                return result;
            }

            if (dataType instanceof FunctionDefinition) {
                FunctionDefinition functionDefinition = (FunctionDefinition) dataType;
                ArrayList<Object> parameters = new ArrayList<>();
                for (ParameterDefinition parameter : functionDefinition.getArguments()) {
                    LinkedHashMap<String, Object> p = new LinkedHashMap<>();
                    p.put("name", parameter.getName());
                    p.put("type", typeRef(parameter.getDataType()));
                    parameters.add(p);
                }
                result.put("kind", "function");
                result.put("return_type", typeRef(functionDefinition.getReturnType()));
                result.put("parameters", parameters);
                return result;
            }

            String baseName = ensureBase(dataType);
            result.put("kind", "base");
            result.put("name", baseName);
            return result;
        }

        private boolean isBaseLike(DataType dataType) {
            return !(dataType instanceof Structure)
                && !(dataType instanceof Union)
                && !(dataType instanceof Enum)
                && !(dataType instanceof TypeDef)
                && !(dataType instanceof Pointer)
                && !(dataType instanceof Array)
                && !(dataType instanceof FunctionDefinition);
        }

        private String ensureBase(DataType dataType) {
            if (dataType == null || dataType instanceof VoidDataType) {
                ensureVoid();
                return "void";
            }

            String name = typeName(dataType);
            String kind = baseKind(dataType);
            int size = Math.max(dataType.getLength(), 0);
            boolean signed = isSignedBase(dataType);

            if (size == 0 && !"void".equals(name)) {
                name = "opaque_0";
            }
            ensureSyntheticBase(name, size, kind, signed);
            return name;
        }

        private void ensureVoid() {
            ensureSyntheticBase("void", 0, "void", false);
        }

        private void ensurePointer() {
            ensureSyntheticBase("pointer", pointerSize, "pointer", false);
        }

        private void ensureSyntheticBase(String name, int size, String kind, boolean signed) {
            if (baseTypes.containsKey(name)) {
                return;
            }
            LinkedHashMap<String, Object> record = new LinkedHashMap<>();
            record.put("size", Math.max(size, 0));
            record.put("signed", signed);
            record.put("kind", kind);
            record.put("endian", endian);
            baseTypes.put(name, record);
        }

        private String baseKind(DataType dataType) {
            String lower = dataType.getName().toLowerCase(Locale.ROOT);
            if (dataType instanceof VoidDataType || dataType.getLength() == 0 || "void".equals(lower)) {
                return "void";
            }
            if (dataType instanceof BooleanDataType || lower.contains("bool")) {
                return "bool";
            }
            if (dataType instanceof CharDataType || lower.equals("char") || lower.endsWith(" char")) {
                return "char";
            }
            if (dataType instanceof FloatDataType || lower.contains("float") || lower.contains("double")) {
                return "float";
            }
            return "int";
        }

        private boolean isSignedBase(DataType dataType) {
            String lower = dataType.getName().toLowerCase(Locale.ROOT);
            if (lower.contains("unsigned") || lower.startsWith("u") || lower.startsWith("uint")
                    || lower.contains("byte")) {
                return false;
            }
            return !"bool".equals(baseKind(dataType)) && !"void".equals(baseKind(dataType));
        }

        private String typeName(DataType dataType) {
            String name = dataType.getName();
            if (name == null || name.isEmpty() || name.startsWith("undefined")) {
                String category = dataType.getCategoryPath() == null
                    ? "root"
                    : dataType.getCategoryPath().getPath();
                name = "unnamed_" + Integer.toHexString((category + ":" + dataType.getPathName()).hashCode());
            }
            return name;
        }
    }

    private static int firstPositiveInt(Object target, int fallback, String... methodNames) {
        for (String methodName : methodNames) {
            try {
                Method method = target.getClass().getMethod(methodName);
                Object value = method.invoke(target);
                if (value instanceof Number) {
                    int intValue = ((Number) value).intValue();
                    if (intValue >= 0) {
                        return intValue;
                    }
                }
            } catch (Exception ignored) {
                // Try the next method name for compatibility across Ghidra versions.
            }
        }
        return fallback;
    }

    private static final class JsonWriter {
        static void write(Object value, Appendable out) throws IOException {
            writeValue(value, out, 0);
        }

        @SuppressWarnings("unchecked")
        private static void writeValue(Object value, Appendable out, int indent) throws IOException {
            if (value == null) {
                out.append("null");
            } else if (value instanceof String) {
                writeString((String) value, out);
            } else if (value instanceof Number || value instanceof Boolean) {
                out.append(String.valueOf(value));
            } else if (value instanceof Map) {
                writeMap((Map<String, Object>) value, out, indent);
            } else if (value instanceof Iterable) {
                writeIterable((Iterable<?>) value, out, indent);
            } else {
                writeString(String.valueOf(value), out);
            }
        }

        private static void writeMap(Map<String, Object> map, Appendable out, int indent) throws IOException {
            out.append('{');
            if (!map.isEmpty()) {
                boolean first = true;
                for (Map.Entry<String, Object> entry : map.entrySet()) {
                    if (!first) {
                        out.append(',');
                    }
                    newline(out, indent + 2);
                    writeString(entry.getKey(), out);
                    out.append(": ");
                    writeValue(entry.getValue(), out, indent + 2);
                    first = false;
                }
                newline(out, indent);
            }
            out.append('}');
        }

        private static void writeIterable(Iterable<?> values, Appendable out, int indent) throws IOException {
            out.append('[');
            boolean first = true;
            for (Object value : values) {
                if (!first) {
                    out.append(',');
                }
                newline(out, indent + 2);
                writeValue(value, out, indent + 2);
                first = false;
            }
            if (!first) {
                newline(out, indent);
            }
            out.append(']');
        }

        private static void newline(Appendable out, int indent) throws IOException {
            out.append('\n');
            for (int i = 0; i < indent; i++) {
                out.append(' ');
            }
        }

        private static void writeString(String value, Appendable out) throws IOException {
            out.append('"');
            for (int i = 0; i < value.length(); i++) {
                char c = value.charAt(i);
                switch (c) {
                    case '"':
                        out.append("\\\"");
                        break;
                    case '\\':
                        out.append("\\\\");
                        break;
                    case '\b':
                        out.append("\\b");
                        break;
                    case '\f':
                        out.append("\\f");
                        break;
                    case '\n':
                        out.append("\\n");
                        break;
                    case '\r':
                        out.append("\\r");
                        break;
                    case '\t':
                        out.append("\\t");
                        break;
                    default:
                        if (c < 0x20) {
                            out.append(String.format("\\u%04x", (int) c));
                        } else {
                            out.append(c);
                        }
                }
            }
            out.append('"');
        }
    }
}
