from flask import Flask, request, render_template
import joblib

# Initialize the Flask application
app = Flask(__name__)

# Load pre-trained model and tfidf vectorizer
model = joblib.load('model.pkl')
tfidf = joblib.load('tfidf.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        review = request.form['review']
        # Vectorize the user input
        review_vector = tfidf.transform([review]) 
        # Predict the sentiment 
        prediction = model.predict(review_vector)  
        result = "Positive" if prediction[0] == 1 else "Negative"
        # Pass the original review text back to the template
        return render_template('index.html', prediction=result, review_text=review)
    else:
        # If it's a GET request, render the form empty
        return render_template('index.html', prediction=None, review_text='')



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

