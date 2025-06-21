import asyncio
import os
from src.utils.config import load_config
from src.models.llm_handler import LLMHandler

async def test_llm_handler():
    config = load_config()
    handler = LLMHandler(config)
    
    print("Testing LLM Handler...")
    print(f"Config loaded: {config.model.name}")
    print(f"Handler created: {handler.is_loaded}")
    
    try:
        await handler.load_model()
        print("✅ Model loaded successfully!")
        
        # Test text generation
        print("\n🤖 Testing text generation...")
        test_prompts = [
            "The benefits of open source software are:",
            "Python is a programming language that:",
            "Machine learning can be defined as:"
        ]
        
        for prompt in test_prompts:
            print(f"\n📝 Prompt: {prompt}")
            result = await handler.generate(prompt, max_tokens=50, temperature=0.7)
            print(f"🤖 Response: {result['text']}")
            print(f"📊 Tokens: {result['tokens_generated']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

# Run the test
asyncio.run(test_llm_handler())