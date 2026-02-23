import argparse
import struct
import sys
from dwarffi.parser import load_isf_json, VtypeJson, _JSON_LIB_USED

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser(
        description="Load and parse a dwarf2json ISF (Intermediate Symbol File) JSON or JSON.XZ.",
        epilog=f"This script uses the '{_JSON_LIB_USED}' library for JSON parsing."
    )
    cli_parser.add_argument("json_file_path", type=str,
                            help="Path to the ISF JSON or JSON.XZ file.")
    cli_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print detailed info.")
    cli_parser.add_argument("--find-symbol-at", type=lambda x: int(x, 0),
                            metavar="ADDRESS", help="Find symbols at address.")
    cli_parser.add_argument(
        "--test-write", action="store_true", help="Run field write test.")
    cli_parser.add_argument(
        "--test-to-bytes", action="store_true", help="Run to_bytes() test.")
    cli_parser.add_argument("--test-base-enum-instance", action="store_true",
                            help="Test creating instances of base/enum types.")
    cli_parser.add_argument(
        "--get-type", type=str, help="Test the generic get_type method with the provided type name.")
    cli_parser.add_argument(
        "--test-array-write", action="store_true", help="Test writing to array elements.")

    args = cli_parser.parse_args()
    print(
        f"Attempting to load ISF file: {args.json_file_path} (using {_JSON_LIB_USED})")

    try:
        isf_data: VtypeJson = load_isf_json(args.json_file_path)
        print("\nSuccessfully loaded ISF JSON.")
        print(f"  ISF Representation: {isf_data}")

        if args.get_type:
            print(f"\n--- Testing get_type('{args.get_type}') ---")
            found_type_obj = isf_data.get_type(args.get_type)
            if found_type_obj:
                print(f"  Found type: {found_type_obj}")
                print(f"  Type class: {found_type_obj.__class__.__name__}")
            else:
                print(f"  Type '{args.get_type}' not found.")

        if args.find_symbol_at is not None:
            print(
                f"\n--- Finding symbols at address {args.find_symbol_at:#x} ---")
            symbols_at_addr = isf_data.get_symbols_by_address(
                args.find_symbol_at)
            if symbols_at_addr:
                print(
                    f"  Found {len(symbols_at_addr)} symbol(s) at {args.find_symbol_at:#x}:")
                for sym_obj in symbols_at_addr:
                    print(f"    - {sym_obj}")
            else:
                print(
                    f"  No symbols found at address {args.find_symbol_at:#x}.")
            if isf_data._address_to_symbol_list_cache is not None:
                print(
                    f"  Address-to-symbol cache is now populated with {len(isf_data._address_to_symbol_list_cache)} entries.")

        if args.verbose:
            print("\n--- Verbose Information ---")
            print(
                f"  Metadata Producer: {isf_data.metadata.producer.get('name', 'N/A')}, Version: {isf_data.metadata.producer.get('version', 'N/A')}")
            print(f"  ISF Format Version: {isf_data.metadata.format_version}")
            print(
                f"  Number of raw base types defined: {len(isf_data._raw_base_types)}")
            print(
                f"  Number of raw user types defined: {len(isf_data._raw_user_types)}")
            print(f"  Number of raw enums defined: {len(isf_data._raw_enums)}")
            print(
                f"  Number of raw symbols defined: {len(isf_data._raw_symbols)}")

        if args.test_write or args.test_to_bytes or args.test_array_write:
            print("\n--- Testing Field Write, To Bytes, and/or Array Write Functionality ---")
            struct_to_test = "my_struct"
            struct_def = isf_data.get_user_type(struct_to_test)

            if struct_def and struct_def.size is not None:
                buffer_data = bytearray(struct_def.size)

                if struct_to_test == "my_struct":
                    struct.pack_into("<i", buffer_data, 0, 100)
                    struct.pack_into("<B", buffer_data, 4, 0b00000001)

                instance = isf_data.create_instance(
                    struct_to_test, buffer_data)

                if args.test_write:
                    print(f"  Testing writes for '{struct_to_test}':")
                    if "id" in struct_def.fields:
                        print(f"    Initial id: {instance.id}")
                        instance.id = 999
                        print(f"    Modified id: {instance.id} (Buffer check: {struct.unpack_from('<i', buffer_data, 0)[0]})")

                if args.test_array_write and "args" in struct_def.fields:
                    print(f"  Testing array writes for '{struct_to_test}.args':")
                    args_array_view = instance.args
                    print(f"    Initial args_array_view[0]: {args_array_view[0] if len(args_array_view) > 0 else 'N/A'}")
                    if len(args_array_view) > 0:
                        args_array_view[0] = 0xAAAAAAAAAAAAAAAA
                        print("    Set args_array_view[0] = 0xAAAAAAAAAAAAAAAA")
                        print(f"    New args_array_view[0]: {args_array_view[0]}")
                    if len(args_array_view) > 1:
                        args_array_view[1] = 0xBBBBBBBBBBBBBBBB
                        print("    Set args_array_view[1] = 0xBBBBBBBBBBBBBBBB")
                        print(f"    New args_array_view[1]: {args_array_view[1]}")

                if args.test_to_bytes:
                    instance_bytes = instance.to_bytes()
                    print(f"  instance.to_bytes() (hex) for '{struct_to_test}': {instance_bytes.hex()}")
            else:
                print(f"  Skipping write/to_bytes/array_write test: '{struct_to_test}' not found or has no size.")

        if args.test_base_enum_instance:
            print("\n--- Testing Base/Enum Instance Creation ---")
            int_def = isf_data.get_base_type("int")
            if int_def and int_def.size is not None:
                int_buffer = bytearray(int_def.size)
                struct.pack_into("<i", int_buffer, 0, 12345)
                int_instance = isf_data.create_instance("int", int_buffer)
                print(f"  Created int_instance: {int_instance}")
                print(f"  int_instance[0]: {int_instance[0]}")
                int_instance[0] = 54321
                print(f"  Modified int_instance[0]: {int_instance[0]}")
                print(f"  int(int_instance): {int(int_instance)}")
                print(f"  int_instance.to_bytes() (hex): {int_instance.to_bytes().hex()}")
            else:
                print("  Skipping 'int' instance test: 'int' not found or has no size.")

    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\nError loading or parsing ISF file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)