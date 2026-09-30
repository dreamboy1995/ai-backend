"""多语言解析验证脚本"""
import tempfile
import os
from app.services.code_index.ast_parser import parse_file


def show(label, code, suffix):
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8") as f:
        f.write(code)
        path = f.name
    try:
        table = parse_file(path)
        print(f"=== {label} ({table.language}) ===")
        for s in sorted(table.symbols, key=lambda x: x.start_line):
            print(f"  {s.symbol_type.value.capitalize()}: {s.name} (Lines {s.start_line}-{s.end_line})")
        print()
    finally:
        os.unlink(path)


# JavaScript
show("JavaScript", '''
const greeting = "hello";

function add(a, b) {
    return a + b;
}

class Calculator {
    constructor() {
        this.value = 0;
    }
    multiply(a, b) {
        return a * b;
    }
}
''', ".js")

# TypeScript
show("TypeScript", '''
const greeting: string = "hello";

function add(a: number, b: number): number {
    return a + b;
}

interface User {
    name: string;
}

class Calculator {
    value: number = 0;
    multiply(a: number, b: number): number {
        return a * b;
    }
}
''', ".ts")

# Go
show("Go", '''
package main

var config = "test"

func add(a, b int) int {
    return a + b
}

type User struct {
    Name string
}

func (u *User) GetName() string {
    return u.Name
}
''', ".go")

# Java
show("Java", '''
public class Hello {
    private int value;

    public Hello(int v) {
        this.value = v;
    }

    public int getValue() {
        return this.value;
    }
}
''', ".java")

print("All language tests completed.")
