from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

def create_and_train_model(data):
    # Clean the data
    data = data.dropna(subset=['Genre', 'Plot'])
    data = data[data['Genre'] != 'unknown']
    
    # Encode genres
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['Genre'])
    
    # Vectorize the plot text
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(data['Plot'])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    return model, vectorizer, label_encoder 