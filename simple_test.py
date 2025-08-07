#!/usr/bin/env python3
"""
Simple test without complex async TaskGroups
"""

from app import InferlessPythonModel, RequestObjects

def simple_test():
    print("🧪 Testing with single query...")
    
    # Initialize the model  
    model = InferlessPythonModel()
    model.initialize()
    
    # Simple single test
    request = RequestObjects(user_query="Find pizza in New York")
    
    try:
        print("🔍 Testing query: Find pizza in New York")
        response = model.infer(request)
        print("✅ SUCCESS!")
        print("🎯 Response:")
        print(response.generated_result)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test()