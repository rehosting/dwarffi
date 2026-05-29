// Imports Volatility-style Intermediate Symbol File (ISF) JSON into the
// current Ghidra program.
//
// Usage from Ghidra Script Manager:
//   Run the script and choose an input .json file when prompted.
//
// Usage from analyzeHeadless:
//   analyzeHeadless /tmp/proj Proj -import sample.bin \
//     -scriptPath /path/to/src/dwarffi/ghidra_scripts \
//     -postScript ISF2Ghidra.java /tmp/sample.isf.json
//
// Optional arguments:
//   --types-only    Import only data types
//   --no-symbols    Do not create labels/data from ISF symbols
//   --no-functions  Do not create/update functions from ISF functions

import ghidra.app.script.GhidraScript;
import ghidra.program.database.function.OverlappingFunctionException;
import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressSet;
import ghidra.program.model.data.AbstractIntegerDataType;
import ghidra.program.model.data.ArrayDataType;
import ghidra.program.model.data.BooleanDataType;
import ghidra.program.model.data.ByteDataType;
import ghidra.program.model.data.CategoryPath;
import ghidra.program.model.data.CharDataType;
import ghidra.program.model.data.DataType;
import ghidra.program.model.data.DataTypeConflictHandler;
import ghidra.program.model.data.DataTypeManager;
import ghidra.program.model.data.DoubleDataType;
import ghidra.program.model.data.EnumDataType;
import ghidra.program.model.data.Float4DataType;
import ghidra.program.model.data.Float8DataType;
import ghidra.program.model.data.FloatDataType;
import ghidra.program.model.data.InvalidDataTypeException;
import ghidra.program.model.data.PointerDataType;
import ghidra.program.model.data.StructureDataType;
import ghidra.program.model.data.TypedefDataType;
import ghidra.program.model.data.UnionDataType;
import ghidra.program.model.data.VoidDataType;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Listing;
import ghidra.program.model.listing.ParameterImpl;
import ghidra.program.model.listing.Variable;
import ghidra.program.model.symbol.SourceType;
import ghidra.program.model.symbol.SymbolTable;
import ghidra.program.model.util.CodeUnitInsertionException;
import ghidra.util.exception.DuplicateNameException;
import ghidra.util.exception.InvalidInputException;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

public class ISF2Ghidra extends GhidraScript {
    private static final CategoryPath ISF_CATEGORY = new CategoryPath("/ISF");

    @Override
    protected void run() throws Exception {
        if (currentProgram == null) {
            printerr("No current program is open.");
            return;
        }

        ImportOptions options = ImportOptions.fromArgs(getScriptArgs());
        if (options.inputFile == null) {
            options.inputFile = askFile("Open ISF JSON", "Import");
        }

        Object parsed = new JsonParser(Files.readString(options.inputFile.toPath(), StandardCharsets.UTF_8)).parse();
        if (!(parsed instanceof Map)) {
            throw new IllegalArgumentException("ISF root must be a JSON object");
        }

        int tx = currentProgram.startTransaction("Import ISF");
        boolean commit = false;
        try {
            IsfImporter importer = new IsfImporter(asMap(parsed), currentProgram.getDataTypeManager());
            importer.importTypes();
            if (options.importSymbols) {
                importer.importSymbols(currentProgram.getListing(), currentProgram.getSymbolTable());
            }
            if (options.importFunctions) {
                importer.importFunctions();
            }
            commit = true;
            println(String.format(
                "Imported ISF from %s (%d base types, %d user types, %d enums, %d typedefs, %d symbols, %d functions)",
                options.inputFile.getAbsolutePath(),
                importer.baseCount,
                importer.userTypeCount,
                importer.enumCount,
                importer.typedefCount,
                importer.symbolCount,
                importer.functionCount
            ));
        } finally {
            currentProgram.endTransaction(tx, commit);
        }
    }

    private static final class ImportOptions {
        File inputFile;
        boolean importSymbols = true;
        boolean importFunctions = true;

        static ImportOptions fromArgs(String[] args) {
            ImportOptions options = new ImportOptions();
            if (args == null) {
                return options;
            }
            for (String arg : args) {
                if ("--types-only".equals(arg)) {
                    options.importSymbols = false;
                    options.importFunctions = false;
                } else if ("--no-symbols".equals(arg)) {
                    options.importSymbols = false;
                } else if ("--no-functions".equals(arg)) {
                    options.importFunctions = false;
                } else if (options.inputFile == null) {
                    options.inputFile = new File(arg);
                } else {
                    throw new IllegalArgumentException("Unexpected argument: " + arg);
                }
            }
            return options;
        }
    }

    private final class IsfImporter {
        private final Map<String, Object> isf;
        private final DataTypeManager dataTypeManager;
        private final Map<String, DataType> baseTypes = new TreeMap<>();
        private final Map<String, DataType> userTypes = new TreeMap<>();
        private final Map<String, DataType> enums = new TreeMap<>();
        private final Map<String, DataType> typedefs = new TreeMap<>();
        private final Set<String> resolvingTypedefs = new HashSet<>();

        int baseCount;
        int userTypeCount;
        int enumCount;
        int typedefCount;
        int symbolCount;
        int functionCount;

        IsfImporter(Map<String, Object> isf, DataTypeManager dataTypeManager) {
            this.isf = isf;
            this.dataTypeManager = dataTypeManager;
        }

        void importTypes() throws Exception {
            importBaseTypes();
            predeclareUserTypes();
            importEnums();
            importTypedefs();
            populateUserTypes();
        }

        private void importBaseTypes() {
            for (Map.Entry<String, Object> entry : objectMap("base_types").entrySet()) {
                Map<String, Object> spec = asMap(entry.getValue());
                DataType dataType = baseDataType(entry.getKey(), spec);
                baseTypes.put(entry.getKey(), dataType);

                if (!isBuiltInName(entry.getKey(), dataType)) {
                    DataType alias = new TypedefDataType(ISF_CATEGORY, entry.getKey(), dataType);
                    dataTypeManager.resolve(alias, DataTypeConflictHandler.REPLACE_HANDLER);
                }
                baseCount += 1;
            }
            baseTypes.putIfAbsent("void", VoidDataType.dataType);
            baseTypes.putIfAbsent("pointer", new PointerDataType(dataTypeManager));
        }

        private DataType baseDataType(String name, Map<String, Object> spec) {
            String kind = stringValue(spec.get("kind"), "int");
            int size = intValue(spec.get("size"), 1);
            boolean signed = boolValue(spec.get("signed"), true);

            if ("void".equals(kind) || size == 0) {
                return VoidDataType.dataType;
            }
            if ("pointer".equals(kind) || "pointer".equals(name)) {
                return new PointerDataType(dataTypeManager);
            }
            if ("bool".equals(kind)) {
                return BooleanDataType.dataType;
            }
            if ("char".equals(kind)) {
                return CharDataType.dataType;
            }
            if ("float".equals(kind)) {
                if (size == 4) {
                    return Float4DataType.dataType;
                }
                if (size == 8) {
                    return Float8DataType.dataType;
                }
                if (size == 8 && name.toLowerCase(Locale.ROOT).contains("double")) {
                    return DoubleDataType.dataType;
                }
                return FloatDataType.dataType;
            }
            DataType integer = signed
                ? AbstractIntegerDataType.getSignedDataType(size, dataTypeManager)
                : AbstractIntegerDataType.getUnsignedDataType(size, dataTypeManager);
            return integer == null ? ByteDataType.dataType : integer;
        }

        private boolean isBuiltInName(String name, DataType dataType) {
            return name.equals(dataType.getName())
                || "void".equals(name)
                || "pointer".equals(name)
                || dataType instanceof PointerDataType;
        }

        private void predeclareUserTypes() {
            for (Map.Entry<String, Object> entry : objectMap("user_types").entrySet()) {
                String name = entry.getKey();
                Map<String, Object> spec = asMap(entry.getValue());
                String kind = stringValue(spec.get("kind"), "struct");
                int size = intValue(spec.get("size"), 0);
                DataType dataType = "union".equals(kind)
                    ? new UnionDataType(ISF_CATEGORY, name, dataTypeManager)
                    : new StructureDataType(ISF_CATEGORY, name, Math.max(size, 0), dataTypeManager);
                userTypes.put(name, dataType);
                userTypeCount += 1;
            }
        }

        private void importEnums() {
            for (Map.Entry<String, Object> entry : objectMap("enums").entrySet()) {
                String name = entry.getKey();
                Map<String, Object> spec = asMap(entry.getValue());
                int size = Math.max(intValue(spec.get("size"), 4), 1);
                EnumDataType enumType = new EnumDataType(ISF_CATEGORY, name, size, dataTypeManager);
                for (Map.Entry<String, Object> constant : asMap(spec.get("constants")).entrySet()) {
                    enumType.add(constant.getKey(), longValue(constant.getValue(), 0));
                }
                enums.put(name, dataTypeManager.resolve(enumType, DataTypeConflictHandler.REPLACE_HANDLER));
                enumCount += 1;
            }
        }

        private void populateUserTypes() throws InvalidDataTypeException {
            for (Map.Entry<String, Object> entry : objectMap("user_types").entrySet()) {
                String name = entry.getKey();
                Map<String, Object> spec = asMap(entry.getValue());
                DataType dataType = userTypes.get(name);
                List<FieldSpec> fields = fields(spec);
                if (dataType instanceof StructureDataType) {
                    StructureDataType structure = (StructureDataType) dataType;
                    structure.deleteAll();
                    setStructureLength(structure, Math.max(intValue(spec.get("size"), 0), 0));
                    for (FieldSpec field : fields) {
                        addStructureField(structure, field);
                    }
                    userTypes.put(name, dataTypeManager.resolve(structure, DataTypeConflictHandler.REPLACE_HANDLER));
                } else if (dataType instanceof UnionDataType) {
                    UnionDataType union = (UnionDataType) dataType;
                    for (FieldSpec field : fields) {
                        addUnionField(union, field);
                    }
                    userTypes.put(name, dataTypeManager.resolve(union, DataTypeConflictHandler.REPLACE_HANDLER));
                }
            }
        }

        private void setStructureLength(StructureDataType structure, int length) {
            try {
                Method setLength = structure.getClass().getMethod("setLength", int.class);
                setLength.invoke(structure, length);
                return;
            } catch (Exception ignored) {
                // Ghidra 11.0 lacks setLength; grow the empty structure instead.
            }
            int delta = length - Math.max(structure.getLength(), 0);
            if (delta > 0) {
                structure.growStructure(delta);
            }
        }

        private List<FieldSpec> fields(Map<String, Object> spec) {
            ArrayList<FieldSpec> result = new ArrayList<>();
            for (Map.Entry<String, Object> fieldEntry : asMap(spec.get("fields")).entrySet()) {
                Map<String, Object> field = asMap(fieldEntry.getValue());
                result.add(new FieldSpec(
                    fieldEntry.getKey(),
                    intValue(field.get("offset"), 0),
                    asMap(field.get("type")),
                    boolValue(field.get("anonymous"), false)
                ));
            }
            result.sort(Comparator.comparingInt((FieldSpec f) -> f.offset).thenComparing(f -> f.name));
            return result;
        }

        private void addStructureField(StructureDataType structure, FieldSpec field)
                throws InvalidDataTypeException {
            Map<String, Object> typeSpec = field.typeSpec;
            String name = field.anonymous ? null : field.name;
            if ("bitfield".equals(stringValue(typeSpec.get("kind"), ""))) {
                DataType baseType = typeFromSpec(asMap(typeSpec.get("type")));
                int bitLength = intValue(typeSpec.get("bit_length"), 0);
                int bitPosition = intValue(typeSpec.get("bit_position"), 0);
                int byteWidth = Math.max(baseType.getLength(), 1);
                structure.insertBitFieldAt(field.offset, byteWidth, bitPosition, baseType, bitLength, name, null);
                return;
            }

            DataType fieldType = typeFromSpec(typeSpec);
            int length = Math.max(fieldType.getLength(), 1);
            structure.insertAtOffset(field.offset, fieldType, length, name, null);
        }

        private void addUnionField(UnionDataType union, FieldSpec field) throws InvalidDataTypeException {
            Map<String, Object> typeSpec = field.typeSpec;
            String name = field.anonymous ? null : field.name;
            if ("bitfield".equals(stringValue(typeSpec.get("kind"), ""))) {
                DataType baseType = typeFromSpec(asMap(typeSpec.get("type")));
                union.addBitField(baseType, intValue(typeSpec.get("bit_length"), 0), name, null);
                return;
            }
            DataType fieldType = typeFromSpec(typeSpec);
            union.add(fieldType, Math.max(fieldType.getLength(), 1), name, null);
        }

        private void importTypedefs() {
            for (Map.Entry<String, Object> entry : objectMap("typedefs").entrySet()) {
                ensureTypedef(entry.getKey());
            }
        }

        private DataType ensureTypedef(String name) {
            DataType existing = typedefs.get(name);
            if (existing != null) {
                return existing;
            }
            if (resolvingTypedefs.contains(name)) {
                return ByteDataType.dataType;
            }
            Object targetSpec = objectMap("typedefs").get(name);
            if (targetSpec == null) {
                return null;
            }

            resolvingTypedefs.add(name);
            DataType target = typeFromSpec(asMap(targetSpec));
            resolvingTypedefs.remove(name);

            DataType typedef = new TypedefDataType(ISF_CATEGORY, name, target, dataTypeManager);
            DataType resolved = dataTypeManager.resolve(typedef, DataTypeConflictHandler.REPLACE_HANDLER);
            typedefs.put(name, resolved);
            typedefCount += 1;
            return resolved;
        }

        void importSymbols(Listing listing, SymbolTable symbolTable) {
            for (Map.Entry<String, Object> entry : objectMap("symbols").entrySet()) {
                Map<String, Object> spec = asMap(entry.getValue());
                Address address = address(spec.get("address"));
                if (address == null) {
                    continue;
                }
                try {
                    if (symbolTable.getGlobalSymbol(entry.getKey(), address) == null) {
                        symbolTable.createLabel(address, entry.getKey(), SourceType.IMPORTED);
                    }
                    if (spec.containsKey("type")) {
                        DataType dataType = typeFromSpec(asMap(spec.get("type")));
                        if (dataType != VoidDataType.dataType && listing.getDataAt(address) == null) {
                            listing.createData(address, dataType);
                        }
                    }
                    symbolCount += 1;
                } catch (InvalidInputException | CodeUnitInsertionException ignored) {
                    // Keep importing the rest of the ISF if one address is invalid or occupied.
                }
            }
        }

        void importFunctions() {
            for (Map.Entry<String, Object> entry : objectMap("functions").entrySet()) {
                Map<String, Object> spec = asMap(entry.getValue());
                Address address = address(spec.get("address"));
                if (address == null) {
                    continue;
                }
                try {
                    Function function = currentProgram.getFunctionManager().getFunctionAt(address);
                    if (function == null) {
                        function = currentProgram.getFunctionManager().createFunction(
                            entry.getKey(),
                            address,
                            new AddressSet(address),
                            SourceType.IMPORTED
                        );
                    }
                    function.setReturnType(typeFromSpec(asMap(spec.get("return_type"))), SourceType.IMPORTED);

                    ArrayList<Variable> parameters = new ArrayList<>();
                    int ordinal = 0;
                    for (Object parameterObject : asList(spec.get("parameters"))) {
                        Map<String, Object> parameter = asMap(parameterObject);
                        String paramName = stringValue(parameter.get("name"), "param_" + ordinal);
                        parameters.add(new ParameterImpl(
                            paramName,
                            typeFromSpec(asMap(parameter.get("type"))),
                            ordinal,
                            currentProgram,
                            SourceType.IMPORTED
                        ));
                        ordinal += 1;
                    }
                    function.replaceParameters(
                        parameters,
                        Function.FunctionUpdateType.DYNAMIC_STORAGE_ALL_PARAMS,
                        true,
                        SourceType.IMPORTED
                    );
                    functionCount += 1;
                } catch (InvalidInputException | DuplicateNameException | OverlappingFunctionException ignored) {
                    // Keep importing remaining functions if this address cannot host a function.
                }
            }
        }

        private DataType typeFromSpec(Map<String, Object> spec) {
            String kind = stringValue(spec.get("kind"), "base");
            String name = stringValue(spec.get("name"), "");

            if ("base".equals(kind)) {
                DataType dataType = typedefs.get(name);
                if (dataType == null) {
                    dataType = dataTypeManager.getDataType(ISF_CATEGORY, name);
                }
                if (dataType == null) {
                    dataType = baseTypes.get(name);
                }
                return dataType == null ? ByteDataType.dataType : dataType;
            }
            if ("typedef".equals(kind)) {
                DataType dataType = typedefs.get(name);
                if (dataType == null) {
                    dataType = ensureTypedef(name);
                }
                if (dataType == null) {
                    dataType = dataTypeManager.getDataType(ISF_CATEGORY, name);
                }
                return dataType == null ? ByteDataType.dataType : dataType;
            }
            if ("struct".equals(kind) || "union".equals(kind)) {
                DataType dataType = userTypes.get(name);
                if (dataType == null) {
                    dataType = dataTypeManager.getDataType(ISF_CATEGORY, name);
                }
                return dataType == null ? ByteDataType.dataType : dataType;
            }
            if ("enum".equals(kind)) {
                DataType dataType = enums.get(name);
                if (dataType == null) {
                    dataType = dataTypeManager.getDataType(ISF_CATEGORY, name);
                }
                return dataType == null ? ByteDataType.dataType : dataType;
            }
            if ("pointer".equals(kind)) {
                return new PointerDataType(typeFromSpec(asMap(spec.get("subtype"))), dataTypeManager);
            }
            if ("array".equals(kind)) {
                DataType subtype = typeFromSpec(asMap(spec.get("subtype")));
                return new ArrayDataType(subtype, intValue(spec.get("count"), 0), Math.max(subtype.getLength(), 1), dataTypeManager);
            }
            if ("function".equals(kind)) {
                return new PointerDataType(VoidDataType.dataType, dataTypeManager);
            }
            return ByteDataType.dataType;
        }

        private Address address(Object value) {
            try {
                return currentProgram.getAddressFactory()
                    .getDefaultAddressSpace()
                    .getAddress(longValue(value, 0));
            } catch (Exception e) {
                return null;
            }
        }

        private Map<String, Object> objectMap(String key) {
            return asMap(isf.get(key));
        }
    }

    private static final class FieldSpec {
        final String name;
        final int offset;
        final Map<String, Object> typeSpec;
        final boolean anonymous;

        FieldSpec(String name, int offset, Map<String, Object> typeSpec, boolean anonymous) {
            this.name = name;
            this.offset = offset;
            this.typeSpec = typeSpec;
            this.anonymous = anonymous;
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asMap(Object value) {
        if (value instanceof Map) {
            return (Map<String, Object>) value;
        }
        return new LinkedHashMap<>();
    }

    private static List<Object> asList(Object value) {
        if (value instanceof List) {
            return (List<Object>) value;
        }
        return new ArrayList<>();
    }

    private static String stringValue(Object value, String fallback) {
        return value == null ? fallback : String.valueOf(value);
    }

    private static int intValue(Object value, int fallback) {
        if (value instanceof Number) {
            return ((Number) value).intValue();
        }
        try {
            return value == null ? fallback : Integer.parseInt(String.valueOf(value));
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    private static long longValue(Object value, long fallback) {
        if (value instanceof Number) {
            return ((Number) value).longValue();
        }
        try {
            return value == null ? fallback : Long.parseLong(String.valueOf(value));
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    private static boolean boolValue(Object value, boolean fallback) {
        if (value instanceof Boolean) {
            return (Boolean) value;
        }
        return value == null ? fallback : Boolean.parseBoolean(String.valueOf(value));
    }

    private static final class JsonParser {
        private final String text;
        private int index;

        JsonParser(String text) {
            this.text = text;
        }

        Object parse() {
            Object value = parseValue();
            skipWhitespace();
            if (index != text.length()) {
                throw error("Unexpected trailing input");
            }
            return value;
        }

        private Object parseValue() {
            skipWhitespace();
            if (index >= text.length()) {
                throw error("Unexpected end of input");
            }
            char c = text.charAt(index);
            if (c == '{') {
                return parseObject();
            }
            if (c == '[') {
                return parseArray();
            }
            if (c == '"') {
                return parseString();
            }
            if (c == 't') {
                consumeLiteral("true");
                return Boolean.TRUE;
            }
            if (c == 'f') {
                consumeLiteral("false");
                return Boolean.FALSE;
            }
            if (c == 'n') {
                consumeLiteral("null");
                return null;
            }
            if (c == '-' || Character.isDigit(c)) {
                return parseNumber();
            }
            throw error("Unexpected character: " + c);
        }

        private Map<String, Object> parseObject() {
            expect('{');
            LinkedHashMap<String, Object> result = new LinkedHashMap<>();
            skipWhitespace();
            if (peek('}')) {
                expect('}');
                return result;
            }
            while (true) {
                String key = parseString();
                skipWhitespace();
                expect(':');
                result.put(key, parseValue());
                skipWhitespace();
                if (peek('}')) {
                    expect('}');
                    return result;
                }
                expect(',');
            }
        }

        private List<Object> parseArray() {
            expect('[');
            ArrayList<Object> result = new ArrayList<>();
            skipWhitespace();
            if (peek(']')) {
                expect(']');
                return result;
            }
            while (true) {
                result.add(parseValue());
                skipWhitespace();
                if (peek(']')) {
                    expect(']');
                    return result;
                }
                expect(',');
            }
        }

        private String parseString() {
            expect('"');
            StringBuilder builder = new StringBuilder();
            while (index < text.length()) {
                char c = text.charAt(index++);
                if (c == '"') {
                    return builder.toString();
                }
                if (c != '\\') {
                    builder.append(c);
                    continue;
                }
                if (index >= text.length()) {
                    throw error("Unterminated escape");
                }
                char escaped = text.charAt(index++);
                switch (escaped) {
                    case '"':
                    case '\\':
                    case '/':
                        builder.append(escaped);
                        break;
                    case 'b':
                        builder.append('\b');
                        break;
                    case 'f':
                        builder.append('\f');
                        break;
                    case 'n':
                        builder.append('\n');
                        break;
                    case 'r':
                        builder.append('\r');
                        break;
                    case 't':
                        builder.append('\t');
                        break;
                    case 'u':
                        if (index + 4 > text.length()) {
                            throw error("Invalid unicode escape");
                        }
                        builder.append((char) Integer.parseInt(text.substring(index, index + 4), 16));
                        index += 4;
                        break;
                    default:
                        throw error("Invalid escape: " + escaped);
                }
            }
            throw error("Unterminated string");
        }

        private Number parseNumber() {
            int start = index;
            if (peek('-')) {
                index += 1;
            }
            while (index < text.length() && Character.isDigit(text.charAt(index))) {
                index += 1;
            }
            boolean isFloating = false;
            if (peek('.')) {
                isFloating = true;
                index += 1;
                while (index < text.length() && Character.isDigit(text.charAt(index))) {
                    index += 1;
                }
            }
            if (peek('e') || peek('E')) {
                isFloating = true;
                index += 1;
                if (peek('+') || peek('-')) {
                    index += 1;
                }
                while (index < text.length() && Character.isDigit(text.charAt(index))) {
                    index += 1;
                }
            }
            String number = text.substring(start, index);
            return isFloating ? Double.parseDouble(number) : Long.parseLong(number);
        }

        private void consumeLiteral(String literal) {
            if (!text.startsWith(literal, index)) {
                throw error("Expected " + literal);
            }
            index += literal.length();
        }

        private boolean peek(char c) {
            return index < text.length() && text.charAt(index) == c;
        }

        private void expect(char c) {
            skipWhitespace();
            if (!peek(c)) {
                throw error("Expected '" + c + "'");
            }
            index += 1;
        }

        private void skipWhitespace() {
            while (index < text.length() && Character.isWhitespace(text.charAt(index))) {
                index += 1;
            }
        }

        private IllegalArgumentException error(String message) {
            return new IllegalArgumentException(message + " at byte " + index);
        }
    }
}
