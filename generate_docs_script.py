import os
import ast

def get_docstring(node):
    doc = ast.get_docstring(node)
    if doc:
        return doc.replace('\n', ' ')
    return 'No description available.'

def get_args(node):
    args = []
    if hasattr(node, 'args') and hasattr(node.args, 'args'):
        for a in node.args.args:
            args.append(a.arg)
    return ', '.join(args)

def parse_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content)
        
        functions = []
        classes = []
        module_doc = ast.get_docstring(tree) or 'No module description.'
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'args': get_args(node),
                    'doc': get_docstring(node)
                })
            elif isinstance(node, ast.ClassDef):
                methods = []
                for m in node.body:
                    if isinstance(m, ast.FunctionDef):
                        methods.append({
                            'name': m.name,
                            'args': get_args(m),
                            'doc': get_docstring(m)
                        })
                classes.append({
                    'name': node.name, 
                    'doc': get_docstring(node),
                    'methods': methods
                })
        return {'module_doc': module_doc, 'functions': functions, 'classes': classes}
    except Exception as e:
        return {'error': str(e)}

def generate_docs(root_dir):
    docs = '# Recommendation System Architecture Documentation\n\n'
    docs += 'This document provides a comprehensive overview of the files, classes, methods, and functions in the recommendation pipeline.\n\n'
    
    for dir_name in ['Business_Meaning', 'Mapping_Engine', 'Model_Engine']:
        dir_path = os.path.join(root_dir, dir_name)
        if not os.path.exists(dir_path):
            continue
            
        docs += f'## Module: {dir_name}\n\n'
        
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    filepath = os.path.join(root, file)
                    rel_path = os.path.relpath(filepath, root_dir).replace(chr(92), '/')
                    info = parse_file(filepath)
                    
                    if 'error' in info:
                        continue
                        
                    if not info.get('functions') and not info.get('classes'):
                        continue
                        
                    docs += f'### File: `{rel_path}`\n\n'
                    docs += f'{info.get("module_doc", "")}\n\n'
                    
                    if info['classes']:
                        for cls in info['classes']:
                            cls_name = cls['name']
                            docs += f'#### Class: `{cls_name}`\n'
                            docs += f'{cls["doc"]}\n\n'
                            if cls['methods']:
                                docs += 'Methods:\n'
                                for method in cls['methods']:
                                    docs += f'- **`{method["name"]}({method["args"]})`**: {method["doc"]}\n'
                            docs += '\n'
                            
                    if info['functions']:
                        docs += '#### Global Functions:\n'
                        for func in info['functions']:
                            docs += f'- **`{func["name"]}({func["args"]})`**: {func["doc"]}\n'
                        docs += '\n'
                        
    with open(os.path.join(root_dir, 'architecture_detailed_docs.md'), 'w', encoding='utf-8') as f:
        f.write(docs)

generate_docs('e:/docker-crash-course/RecSys')
