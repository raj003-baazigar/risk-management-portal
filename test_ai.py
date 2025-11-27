import google.generativeai as genai

KEY = "AIzaSyCHl2ic200rooLyFOScbf5II6rFu9y-8QQ" # Your key
genai.configure(api_key=KEY)

print("Testing API connection...")

try:
    # UPDATED MODEL NAME: gemini-1.5-flash
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Hello! Are you online?")
    print("✅ Success! AI said:", response.text)
except Exception as e:
    print("❌ Error:", e)
