"""Compare CUDA ELF images allowing only internal array symbol numbering."""

import hashlib
import re
import struct


NAMING_SOURCE_SHA256 = (
    "d12039a3e788a0644667845fd5f5cc3cc904be2f01bc630fb19173c3a2f7e701"
)
HEADER = struct.Struct("<16sHHIQQQIHHHHHH")
SECTION = struct.Struct("<IIQQQQIIQQ")
PROGRAM = struct.Struct("<IIQQQQQQ")
SYMBOL = struct.Struct("<IBBHQQ")


def require(condition, reason):
    """Refuse an unproved difference instead of normalizing it."""
    if not condition:
        raise ValueError(reason)


def strings(data):
    """Decode complete, ordered, null-terminated ELF string entries."""
    require(data[:1] == b"\0" and data[-1:] == b"\0", "String framing")
    values, offsets, cursor = [], [], 0
    for value in data.split(b"\0")[:-1]:
        values.append(value.decode("ascii"))
        offsets.append(cursor)
        cursor += len(value) + 1
    return values, offsets


def parse(data):
    """Parse the narrowly admitted little-endian ELF64 CUDA layout."""
    require(len(data) >= HEADER.size, "Truncated ELF")
    header = HEADER.unpack_from(data)
    require(header[0][:7] == b"\x7fELF\x02\x01\x01", "ELF64 LE version")
    require(header[1:4] == (2, 190, 1), "CUDA executable ELF identity")
    require(
        header[8] == HEADER.size
        and header[9] == PROGRAM.size
        and header[11] == SECTION.size,
        "ELF record widths",
    )
    require(header[10] > 0 and 0 < header[13] < header[12], "ELF inventory")
    sections = [
        SECTION.unpack_from(data, header[6] + i * SECTION.size)
        for i in range(header[12])
    ]
    require(sections[0] == (0,) * 10, "Null section")
    names = sections[header[13]]
    name_data = data[names[4] : names[4] + names[5]]
    names_list, names_offsets = strings(name_data)
    name_map = dict(zip(names_offsets, names_list))
    rows, regions = [], [(0, HEADER.size, "header", 1)]
    for index, section in enumerate(sections):
        if index == 0:
            rows.append(dict(fields=section, name="", payload=b""))
            continue
        require(section[0] in name_map, "Section name boundary")
        require(
            section[1] != 8, "NOBITS layout is not this observed image form"
        )
        offset, size, alignment = section[4], section[5], section[8]
        require(
            alignment > 0 and alignment & (alignment - 1) == 0,
            "Section alignment",
        )
        require(
            offset >= HEADER.size and offset + size <= len(data),
            "Section extent",
        )
        rows.append(
            dict(
                fields=section,
                name=name_map[section[0]],
                payload=data[offset : offset + size],
            )
        )
        if size:
            regions.append(
                (offset, offset + size, f"section:{index}", alignment)
            )
    programs = [
        PROGRAM.unpack_from(data, header[5] + i * PROGRAM.size)
        for i in range(header[10])
    ]
    regions.extend(
        [
            (header[6], header[6] + header[12] * SECTION.size, "sections", 8),
            (header[5], header[5] + header[10] * PROGRAM.size, "programs", 8),
        ]
    )
    regions.sort()
    cursor = 0
    for start, end, label, alignment in regions:
        require(
            start == (cursor + alignment - 1) // alignment * alignment,
            "Noncanonical overlap or extra file gap: " + label,
        )
        require(not any(data[cursor:start]), "Nonzero alignment padding")
        require(end <= len(data), "ELF metadata outside file")
        cursor = end
    require(cursor == len(data), "Trailing bytes outside typed ELF inventory")
    return dict(
        header=header, sections=rows, programs=programs, regions=regions
    )


def compare_cubins(original, fresh, naming_source_sha256):
    """Require equal code/data/relocations and prove local-name-only drift."""
    require(
        naming_source_sha256 == NAMING_SOURCE_SHA256,
        "Installed internal symbol naming source differs",
    )
    result = dict(
        original_sha256=hashlib.sha256(original).hexdigest(),
        fresh_sha256=hashlib.sha256(fresh).hexdigest(),
        raw_bytes_equal=original == fresh,
    )
    if original == fresh:
        return dict(result, admitted=True, identity="exact_cubin_bytes")
    left, right = parse(original), parse(fresh)
    require(
        left["header"][:5] + left["header"][7:]
        == right["header"][:5] + right["header"][7:],
        "ELF header differs beyond file table offsets",
    )
    require(
        [row[2] for row in left["regions"]]
        == [row[2] for row in right["regions"]],
        "File region order",
    )
    a, b = left["sections"], right["sections"]
    require(
        [x["name"] for x in a] == [x["name"] for x in b],
        "Section names or indices differ",
    )
    symbol_indices = [i for i, x in enumerate(a) if x["name"] == ".symtab"]
    require(len(symbol_indices) == 1, "Exactly one symbol table required")
    sym = symbol_indices[0]
    string_index = a[sym]["fields"][6]
    require(a[string_index]["name"] == ".strtab", "Symbol string table link")
    require(a[sym]["fields"][9] == SYMBOL.size, "Symbol record width")
    old_names, old_offsets = strings(a[string_index]["payload"])
    new_names, new_offsets = strings(b[string_index]["payload"])
    require(len(old_names) == len(new_names), "String table inventory")
    old_map = dict(zip(old_offsets, old_names))
    new_map = dict(zip(new_offsets, new_names))
    old_symbols = list(SYMBOL.iter_unpack(a[sym]["payload"]))
    new_symbols = list(SYMBOL.iter_unpack(b[sym]["payload"]))
    require(len(old_symbols) == len(new_symbols), "Symbol inventory")
    for symbols, table in ((old_symbols, old_map), (new_symbols, new_map)):
        generated = [
            table[symbol[0]]
            for symbol in symbols
            if symbol[0] in table
            and re.fullmatch(r"constant_array_[0-9]+", table[symbol[0]])
        ]
        require(
            len(generated) == len(set(generated)),
            "Generated constant-array symbol names collide",
        )
    offset_map = dict(zip(old_offsets, new_offsets))
    renames = {}
    records = []
    for index, (old, new) in enumerate(zip(old_symbols, new_symbols)):
        require(old[1:] == new[1:], "Symbol metadata/value/size differs")
        require(
            old[0] in old_map and new[0] in new_map, "Symbol name boundary"
        )
        require(
            new[0] == offset_map[old[0]],
            "Symbol string-entry identity differs",
        )
        before, after = old_map[old[0]], new_map[new[0]]
        if before != after:
            require(
                re.fullmatch(r"constant_array_[0-9]+", before)
                and re.fullmatch(r"constant_array_[0-9]+", after),
                "Only installed internal constant-array names may differ",
            )
            require(
                old[1] == 1 and old[2] == 0,
                "Renamed symbol must be local object with default visibility",
            )
            require(
                0 < old[3] < len(a) and old[5] > 0,
                "Renamed symbol must have a concrete object extent",
            )
            section = a[old[3]]
            require(
                section["name"].startswith(".nv.constant")
                and section["fields"][2] & 2
                and old[4] + old[5] <= section["fields"][5],
                "Renamed object must stay inside identical constant data",
            )
            require(
                before not in renames or renames[before] == after,
                "Ambiguous symbol renaming",
            )
            renames[before] = after
            records.append(
                dict(
                    symbol_index=index,
                    original_name=before,
                    fresh_name=after,
                    section=section["name"],
                    value=old[4],
                    size=old[5],
                )
            )
    require(
        renames and len(set(renames.values())) == len(renames),
        "Internal names must have a bijective renaming",
    )
    require(
        [renames.get(name, name) for name in old_names] == new_names,
        "String bytes differ beyond the admitted symbol names",
    )
    for i, (old, new) in enumerate(zip(a, b)):
        fields_a, fields_b = list(old["fields"]), list(new["fields"])
        fields_a[4] = fields_b[4] = 0
        if i == string_index:
            fields_a[5] = fields_b[5] = 0
        require(
            fields_a == fields_b, "Section metadata differs: " + old["name"]
        )
        if i not in (sym, string_index):
            require(
                old["payload"] == new["payload"],
                "Section payload differs: " + old["name"],
            )

    def anchors(image, offset, ending):
        return {
            label
            for start, end, label, _ in image["regions"]
            if (end if ending else start) == offset
        }

    segment_records = []
    for i, (old, new) in enumerate(zip(left["programs"], right["programs"])):
        require(
            old[:2] + old[3:5] + old[7:] == new[:2] + new[3:5] + new[7:],
            "Program type/flags/addresses/alignment differ",
        )
        require(
            old[6] == old[5] and new[6] == new[5],
            "Program memory extent is not its file-backed extent",
        )
        start = anchors(left, old[2], False) & anchors(right, new[2], False)
        end = anchors(left, old[2] + old[5], True) & anchors(
            right, new[2] + new[5], True
        )
        require(start and end, "Program span changed logical ELF boundaries")
        segment_records.append(
            dict(
                index=i,
                original=list(old),
                fresh=list(new),
                start_anchors=sorted(start),
                end_anchors=sorted(end),
            )
        )
    return dict(
        result,
        admitted=True,
        identity="section_bound_local_symbol_renumbering",
        symbol_renames=records,
        programs=segment_records,
        original_layout=left["regions"],
        fresh_layout=right["regions"],
        unchanged_section_payloads=[
            x["name"] for i, x in enumerate(a) if i not in (sym, string_index)
        ],
    )
