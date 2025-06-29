# Romance Classifier

This project is a simple n-gram based language classifier that identifies inputted text in Romance languages and English. The neural net is built with PyTorch and deployed with Streamlit

# Use

Type in a sentence in Spanish, French, Portuguese, Italian, Romanian, or English, and the model (hopefully) predicts what lanugage the sentence is in. The model is also trained to recognize unknown languages, in which case it will output "Unknown". 

# How it Works

The file *data_loader.py* retrieves 200 sentences from each of the topics  "History", "Computer Science", and "Earth" for each language. The file also takes 300 sentences randomly selected from 1000 Parallel TedEd talks in each language. Providing the model with a mix of formal and semi-conversational text increases its ability to reconize most registers and nuaces of each langauge, but the model still has a lack of recognition of very colloqial and slangy vocabulary. 


# Future improvements + Issues

To improve the data accuracy, I would expand the neural net by ading more hidden neurons, increase data size and quality. 

## Try it

**Deployed** 
Visit: https://language-classifier.streamlit.app/

**Locally**

# Clone and set up
git clone https://github.com/evan_thoms/romance-classifier.git
cd romance-classifier
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the app
streamlit run src/streamlit.py