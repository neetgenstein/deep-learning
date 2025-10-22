import requests
import json

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

SECRET = 'AIzaSyC8dnV15iSJJ7HlGERiQORaCiXVV5GGfoI'

def get_response(prompt):
    headers = {
        'Content-Type': 'application/json',
    }
    
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 50000,
        }
    }
    
    try:
        response = requests.post(
            f"{API_URL}?key={SECRET}",
            headers=headers,
            json=data
        )
        
        response.raise_for_status()
        
        result = response.json()
        if 'candidates' in result and result['candidates']:
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return "Error: No response generated. Please try again."
            
    except requests.exceptions.RequestException as e:
        return f"Error making API request: {str(e)}"
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return f"Error processing API response: {str(e)}"

def main():
    print("DL Models (type 'exit' to quit)")
    
    while True:
        user_input = input("\nInput: ")
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Program Exited!")
            break
            
        response = get_response(user_input)
        print("\nOutput:", response)

if __name__ == "__main__":
    main()