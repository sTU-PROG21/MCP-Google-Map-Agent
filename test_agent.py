#!/usr/bin/env python3
"""
Test script for the Google Maps Agent
Run this to test your agent with mock data
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from app import InferlessPythonModel, RequestObjects

def test_agent():
    print("🤖 Initializing Google Maps Agent...")
    
    # Initialize the model
    model = InferlessPythonModel()
    model.initialize()
    
    print("\n✅ Agent initialized successfully!")
    print("\n" + "="*60)
    
    # Test queries
    test_queries = [
        "Can you find me Tea shop in HSR Layout Bangalore with good number of reviews?",
        "Find me pizza places in Manhattan with great ratings",
        "Show me coffee shops in downtown Seattle"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test Query {i}: {query}")
        print("-" * 50)
        
        # Create request
        request = RequestObjects(user_query=query)
        
        # Get response
        try:
            response = model.infer(request)
            print("🎯 Agent Response:")
            print(response.generated_result)
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "="*60)
    
    print("\n🎉 Testing complete!")
    print("\n💡 To switch to real Google Maps API:")
    print("   1. Set DEMO_MODE = False in app.py")
    print("   2. Set your GOOGLE_MAPS_API_KEY environment variable")

if __name__ == "__main__":
    test_agent()