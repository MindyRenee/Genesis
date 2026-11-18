#!/usr/bin/env python3
import sys

sys.path.insert(0, "/home/mindyrenee/.local/share/genesis/projects/rust/src")

from rust.main import ENTRIES, RELATIONS, definition_of, neighbors

print("=== All entries ===")
for e in ENTRIES:
    print(f"{e.name}: {e.definition} (confidence: {e.confidence:.2f}, origin: {e.origin})")

print("\n=== Relations ===")
for src, rels in RELATIONS.items():
    print(f"{src}: {rels}")

print("\n=== Query examples ===")
print("rust:", definition_of("rust"))
print("code:", definition_of("code"))
print("genesis:", definition_of("genesis"))
print("\nNeighbors of rust:", neighbors("rust"))
