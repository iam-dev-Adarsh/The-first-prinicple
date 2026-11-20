import math
from collections import defaultdict

TRAIN = [
    ("Boom shaka shaka laka laka boom boom", 1),
    ("I am maximus decimus meridius boom boom",1),
    ("Baby stand ready for my conquest I am Alexander the great", 1),
    ("I am mansa musa the gold lord the controller of the trade route", 1),
    ("limited fee trail register now" , 1),

    ("meeting notice tommorow discuss in afternoon", 0),
    ("veni vidi vici going in the bg i am the luigi spamming from the fiji", 0),
    ("i came i saw i went", 0),
    ("download report document view attachment", 0),
    ("project report complete submi review", 0),
]

class NaivBayes:
    def fit(self,data):
        self.classes = list({label for _, label in data})
        self.vocab = list({w for text, _ in data for w in text.split()})
        V = len(self.vocab)

        self.log_prior = {}
        self.log_lik = {}

        for cls in self.classes:
            cls_docs = [(text,label) for text, label in data if label == cls]
            self.log_prior[cls] = math.log(len(cls_docs) / len(data))
            
            all_words = [ w for text, _ in cls_docs for w in text.split()]
            word_count = defaultdict(int)
            for w in all_words: word_count[w] += 1
            total  = len(all_words) + V

            self.log_lik[cls] = {w: math.log((word_count[w] + 1) / total) for w in self.vocab}
            self.log_lik[cls]['<UNK>'] = math.log( 1 / total)
        return self

    def predict(self,text):
        words = text.split()
        scores = {}
        for cls in self.classes:
            scores[cls] = self.log_prior[cls]
            for w in words:
                scores[cls] += self.log_lik[cls].get(w, self.log_lik[cls]['<UNK>'])
        pred = max(scores, key=scores.get)

        #Softmax
        max_s = max(scores.values())
        exp_s = {c : math.exp(s - max_s) for c, s in scores.items()}
        total = sum(exp_s.values())
        probs = {c:exp_s[c] / total for c in self.classes}
        return pred, probs 

model = NaivBayes().fit(TRAIN)

#--------------------TEsting--------------------------------
tests = [
    ("shaka shaka boom boom", 1),
    ("maximus decimus meridius",0),
    ("Baby stand ready for arrival",1),
    ("project complete review", 0),
]

print("============Hand written NaivBayes================")
correct = 0
for text, true_label in tests:
    pred, probs = model.predict(text)
    ok = pred == true_label
    if ok: correct += 1
    status = "success!" if ok else "failure!"
    print(f"{status} '{text}' ")
    print(f" P(spam) = {probs[1]:.3f}, P(ham) = {probs[0]:.3f} -> {'spam' if pred == 1 else 'ham'}")

print(f"\nAccuracy: {correct}/ {len(tests)}")


#------------sklearn---------------------------------
print("=========sklearn MultiNomial NB===============================")
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

texts = [t for t, _ in TRAIN]
labels = [l for _, l in TRAIN]

pipe = Pipeline([
    ('vec', CountVectorizer()),
    ('nb', MultinomialNB(alpha=1.0)),
])
pipe.fit(texts, labels)

test_texts = [ t for t, _ in tests]
preds = pipe.predict(test_texts)
probas = pipe.predict_proba(test_texts)

for (text, true_label), pred, prob in zip(tests, preds, probas):
    ok = pred == true_label
    print(f"{1 if ok else 0} '{text}'  P(spam)= {prob[1]:.3f} -> {'spam' if pred==1 else 'ham'}")



# end


    