import os
from google import genai

def main():
    # Attempt to load the API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("WARNING: GEMINI_API_KEY is not set.")
        return

    client = genai.Client(api_key=api_key)

    models_data = []
    
    print("Fetching models...")
    for m in client.models.list():
        # Extrapolate available info
        name = m.name
        version = getattr(m, 'version', 'N/A')
        display_name = getattr(m, 'display_name', 'N/A')
        methods = getattr(m, 'supported_generation_methods', [])
        
        # We can try to extract tokens limit if present
        input_token_limit = getattr(m, 'input_token_limit', 'N/A')
        output_token_limit = getattr(m, 'output_token_limit', 'N/A')
        
        models_data.append({
            "name": name,
            "display": display_name,
            "version": version,
            "methods": len(methods),
            "input_tokens": input_token_limit,
            "output_tokens": output_token_limit
        })
    
    # Sort them generically by name since rate limits/pricing aren't programmatically returned by the API
    models_data.sort(key=lambda x: x['name'])

    print("-" * 110)
    print(f"{'Model Name':<35} | {'Version':<15} | {'Display Name':<25} | {'Input Limit':<12} | {'Output Limit':<12}")
    print("-" * 110)
    for data in models_data:
        print(f"{data['name']:<35} | {str(data['version']):<15} | {str(data['display']):<25} | {str(data['input_tokens']):<12} | {str(data['output_tokens']):<12}")
    print("-" * 110)
    
if __name__ == '__main__':
    main()
