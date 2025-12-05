"""
Integration validation script for IC-Light enhancements
This script validates that all components are properly integrated
"""

import os
import sys

def validate_files():
    """Validate that all required files exist"""
    print("Validating file structure...")
    
    required_files = [
        'gradio_demo.py',
        'gradio_demo_bg.py',
        'gpt_recommendations.py',
        'upscaler.py',
        'static/light_pointer.js',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
            print(f"  ❌ Missing: {file}")
        else:
            print(f"  ✓ Found: {file}")
    
    if missing_files:
        print(f"\n❌ Validation failed: {len(missing_files)} files missing")
        return False
    
    print("\n✓ All required files present")
    return True


def validate_imports():
    """Validate that all modules can be imported"""
    print("\nValidating module imports...")
    
    modules = [
        ('gpt_recommendations', 'GPTRecommendationClient'),
        ('upscaler', 'ImageUpscaler'),
    ]
    
    failed_imports = []
    for module_name, class_name in modules:
        try:
            module = __import__(module_name)
            if hasattr(module, class_name):
                print(f"  ✓ {module_name}.{class_name}")
            else:
                print(f"  ❌ {module_name}.{class_name} not found")
                failed_imports.append(f"{module_name}.{class_name}")
        except ImportError as e:
            print(f"  ❌ Failed to import {module_name}: {e}")
            failed_imports.append(module_name)
    
    if failed_imports:
        print(f"\n❌ Import validation failed: {len(failed_imports)} imports failed")
        return False
    
    print("\n✓ All modules can be imported")
    return True


def validate_css_consistency():
    """Validate that CSS is consistent across demo files"""
    print("\nValidating CSS consistency...")
    
    with open('gradio_demo.py', 'r', encoding='utf-8') as f:
        demo_content = f.read()
    
    with open('gradio_demo_bg.py', 'r', encoding='utf-8') as f:
        demo_bg_content = f.read()
    
    # Check for custom_css presence
    if 'custom_css = """' in demo_content and 'custom_css = """' in demo_bg_content:
        print("  ✓ Both files have custom CSS")
    else:
        print("  ❌ Custom CSS missing in one or both files")
        return False
    
    # Check for key CSS classes
    css_classes = [
        '.gradio-container',
        '.gr-button',
        '.ai-recommend-btn',
        '.gr-image',
        '.gr-gallery'
    ]
    
    for css_class in css_classes:
        if css_class in demo_content and css_class in demo_bg_content:
            print(f"  ✓ {css_class} present in both files")
        else:
            print(f"  ❌ {css_class} missing in one or both files")
            return False
    
    print("\n✓ CSS is consistent across files")
    return True


def validate_features():
    """Validate that all features are implemented"""
    print("\nValidating feature implementation...")
    
    features = {
        'gradio_demo.py': [
            'GPTRecommendationClient',
            'ImageUpscaler',
            'get_ai_recommendations',
            'apply_ai_suggestion',
            'upscaler.upscale_to_1080p',
            'LightPointerControl'
        ],
        'gradio_demo_bg.py': [
            'GPTRecommendationClient',
            'ImageUpscaler',
            'get_ai_recommendations',
            'apply_ai_suggestion',
            'upscaler.upscale_to_1080p',
            'show_background_upload'
        ]
    }
    
    all_present = True
    for file, feature_list in features.items():
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"\n  Checking {file}:")
        for feature in feature_list:
            if feature in content:
                print(f"    ✓ {feature}")
            else:
                print(f"    ❌ {feature} missing")
                all_present = False
    
    if all_present:
        print("\n✓ All features implemented")
        return True
    else:
        print("\n❌ Some features missing")
        return False


def main():
    """Run all validation checks"""
    print("=" * 60)
    print("IC-Light Enhancement Integration Validation")
    print("=" * 60)
    
    results = []
    
    # Run validations
    results.append(("File Structure", validate_files()))
    results.append(("Module Imports", validate_imports()))
    results.append(("CSS Consistency", validate_css_consistency()))
    results.append(("Feature Implementation", validate_features()))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    for check_name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{check_name:.<40} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL VALIDATIONS PASSED")
        print("The IC-Light enhancements are properly integrated!")
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("Please review the errors above and fix the issues.")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
