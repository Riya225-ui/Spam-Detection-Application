import pandas as pd
import nltk
import pickle
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

nltk.download('punkt', quiet=True)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    return ' '.join([ps.stem(w) for w in nltk.word_tokenize(text) if w.isalnum()])

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.iloc[:5572][['v1', 'v2']].copy()
df.columns = ['label', 'text']
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

df['label_enc'] = df['label'].map({'ham': 0, 'spam': 1})
df['transformed_text'] = df['text'].apply(transform_text)

tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['label_enc'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

mnb = MultinomialNB()
mnb.fit(X_train, y_train)

y_pred = mnb.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
    
with open('model.pkl', 'wb') as f:
    pickle.dump(mnb, f)
