import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

def create_local_dataset():
    data = [
        ("The rocket launched into orbit carrying a satellite payload to explore space.", 0),
        ("Astronauts on the international space station conduct zero gravity science experiments.", 0),
        ("NASA plans a moon mission to study lunar surface geology and orbit trajectory.", 0),
        ("Telescope images reveal distant galaxies, black holes, and cosmic radiation.", 0),
        ("Planetary rover discovered signs of water on Mars surface.", 0),
        ("The pitcher struck out three batters in the ninth inning to win the game.", 1),
        ("Home run over the outfield wall secured a victory for the home baseball team.", 1),
        ("Major league baseball players reported to spring training field.", 1),
        ("The catcher caught the foul pop fly behind home plate for an out.", 1),
        ("Shortstop made an incredible diving play to throw the runner out at first base.", 1),
        ("3D render pipeline processes vertices and textures using shader GPU programs.", 2),
        ("Ray tracing algorithms simulate light reflections and shadows in computer graphics.", 2),
        ("Vector graphics scale infinitely without resolution loss or pixelation.", 2),
        ("OpenGL and DirectX libraries are used for rendering interactive 3D graphics.", 2),
        ("Image processing algorithms apply spatial filters for edge detection in images.", 2),
        ("The senate passed new legislation after extensive political debate and voting.", 3),
        ("Government policies regarding national tax regulations sparked congressional discussion.", 3),
        ("Political candidates presented campaign platforms during the public election debate.", 3),
        ("Voters cast ballots in national elections to select representative leaders.", 3),
        ("Diplomatic foreign policy negotiations aimed to resolve international border disputes.", 3)
    ] * 20
    texts, labels = zip(*data)
    return list(texts), list(labels)

def main():
    print("Loading local text dataset...")
    X, y = create_local_dataset()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    mnb = MultinomialNB()
    mnb.fit(X_train_vec, y_train)
    pred_mnb = mnb.predict(X_test_vec)

    bnb = BernoulliNB()
    bnb.fit(X_train_vec, y_train)
    pred_bnb = bnb.predict(X_test_vec)

    print("Multinomial NB Accuracy:", accuracy_score(y_test, pred_mnb))
    print("Bernoulli NB Accuracy:", accuracy_score(y_test, pred_bnb))

if __name__ == "__main__":
    main()
