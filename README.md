
# Romance Classifier

This project is a simple n-gram based language classifier that identifies inputted text in Romance languages and English. The neural net is built with PyTorch and deployed with Streamlit

---

## Try it Now
Visit the deployed app: [https://language-classifier.streamlit.app/](https://language-classifier.streamlit.app/)

---

# How it Works

- The model retrieves 200 sentences from each of the topics  "History", "Computer Science", and "Earth" for each language
- Also takes 300 sentences randomly selected from 1000 Parallel TedEd talks and 100 common phrases for each language.  
- This mix of formal and semi-conversational text increases its ability to reconize most registers and nuaces of each langauge, but the model still has a lack of recognition of very colloqial and slangy vocabulary. 

---

# Accuracy and Improvements

- Test accuracy: ~96 for recognizing the 6 languages and unknown languages. 
- Planned Improvements: increasing neural net size and adding a larger range of slangy and colloqial language data, refrain from using symbol and formatting ridden Wikipedia articles to prevent wasting training power, and sentence splitting by using nltk.

---

# Set Up Locally
```bash
git clone https://github.com/evan_thoms/romance-classifier.git
cd romance-classifier
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the app
streamlit run src/streamlit.py
