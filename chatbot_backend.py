import re
import nltk

from nltk.corpus import brown, stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. DOWNLOAD RESOURCES
# =========================================================

RESOURCES = [
    "brown",
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger_eng"
]

for resource in RESOURCES:
    nltk.download(resource, quiet=True)


# =========================================================
# 2. INITIALIZATION
# =========================================================

STOP_WORDS = set(stopwords.words("english"))

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

vectorizer = None
corpus_matrix = None
corpus_sentences = []

genre_indexes = {}


# =========================================================
# 3. TEXT NORMALIZATION
# =========================================================

def normalize_text(text):

    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================================================
# 4. TOKENIZATION
# =========================================================

def tokenize_text(text):
    return word_tokenize(text)


# =========================================================
# 5. STOPWORD REMOVAL
# =========================================================

def remove_stopwords(tokens):

    return [
        word for word in tokens
        if word not in STOP_WORDS and word.isalnum()
    ]


# =========================================================
# 6. POS TAGGING
# =========================================================

def get_wordnet_pos(tag):

    if tag.startswith("J"):
        return wordnet.ADJ

    elif tag.startswith("V"):
        return wordnet.VERB

    elif tag.startswith("R"):
        return wordnet.ADV

    return wordnet.NOUN


# =========================================================
# 7. STEMMING AND LEMMATIZATION
# =========================================================

def stem_words(tokens):

    return [stemmer.stem(word) for word in tokens]


def lemmatize_words(tokens):

    tagged = pos_tag(tokens)

    return [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged
    ]


# =========================================================
# 8. NLP PIPELINE
# =========================================================

def process_text(text):

    normalized = normalize_text(text)

    tokens = tokenize_text(normalized)

    filtered = remove_stopwords(tokens)

    tagged = pos_tag(tokens)

    stemmed = stem_words(filtered)

    lemmatized = lemmatize_words(filtered)

    return {
        "Normalized Text": normalized,
        "Tokens": tokens,
        "Stopword Removal": filtered,
        "POS Tagging": tagged,
        "Stemming": stemmed,
        "Lemmatization": lemmatized
    }


# =========================================================
# 9. WORDNET DEFINITIONS
# =========================================================

def get_definition(word):

    synsets = wordnet.synsets(word)

    if synsets:
        return synsets[0].definition()

    return None


# =========================================================
# 10. FULL BROWN CORPUS INDEX
# =========================================================

def build_corpus_index():

    global vectorizer, corpus_matrix, corpus_sentences

    if corpus_matrix is not None:
        return

    print("Loading full Brown Corpus...")

    sentences = []

    for sentence in brown.sents():

        text = normalize_text(" ".join(sentence))

        # Ignore very short and excessively long sentences
        word_count = len(text.split())

        if 6 <= word_count <= 40:
            sentences.append(text)

    corpus_sentences = sentences

    print("Total sentences indexed:", len(corpus_sentences))
    print("Building TF-IDF index...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    corpus_matrix = vectorizer.fit_transform(corpus_sentences)

    print("Brown Corpus index ready:", corpus_matrix.shape)


# =========================================================
# 11. QUERY CLEANING (ONLY ONE VERSION)
# =========================================================

QUERY_FILLER = {
    "what", "which", "who", "whom", "whose",
    "when", "where", "why", "how",
    "is", "are", "was", "were", "be", "been",
    "do", "does", "did",
    "can", "could", "would", "should",
    "will", "shall", "may", "might",
    "the", "a", "an", "and", "or", "but",
    "to", "of", "in", "on", "at", "for",
    "from", "with", "by", "into",
    "as", "than", "then",
    "me", "my", "you", "your", "we", "our",
    "they", "their", "it", "this", "that",
    "tell", "show", "find", "give", "describe",
    "explain", "discuss", "mention",
    "sentence", "sentences", "text", "texts",
    "word", "words", "example", "examples",
    "retrieve", "search", "display", "provide",
    "fetch", "list", "some", "related",
    "brown", "corpus", "language", "used", "use",
    "using", "about", "topic", "topics"
}


def clean_retrieval_query(text):

    normalized = normalize_text(text)

    genre = detect_genre(normalized)

    tokens = word_tokenize(normalized)

    keywords = [
        word for word in tokens
        if word.isalpha()
        and word not in QUERY_FILLER
        and word not in STOP_WORDS
    ]

    if keywords:

        tagged = pos_tag(keywords)

        keywords = [
            lemmatizer.lemmatize(word, get_wordnet_pos(tag))
            for word, tag in tagged
        ]

    # Remove genre terms only if a genre is detected
    if genre:

        genre_words = set(genre.replace("_", " ").split())

        for alias in GENRE_ALIASES.get(genre, []):

            genre_words.update(normalize_text(alias).split())

        keywords = [
            word for word in keywords
            if word not in genre_words
        ]

    query = " ".join(keywords).strip()

    # If the user asks only for a genre, retain that genre
    # instead of returning an empty query
    if not query and genre:
        query = genre.replace("_", " ")

    return query, genre


# =========================================================
# 12. BROWN CORPUS GENRES
# =========================================================

GENRE_ALIASES = {
    "romance": [
        "romance", "romantic", "love story", "love stories"
    ],

    "mystery": [
        "mystery", "detective", "crime fiction"
    ],

    "science_fiction": [
        "science fiction", "sci fi", "sci-fi", "scifi"
    ],

    "adventure": [
        "adventure", "western"
    ],

    "humor": [
        "humor", "humour", "funny", "comedy"
    ],

    "fiction": [
        "general fiction", "fiction"
    ]
}


CATEGORY_DESCRIPTIONS = {
    "news": "News reporting and current events",
    "editorial": "Editorial opinions and commentary",
    "reviews": "Reviews and criticism",
    "religion": "Religious writing",
    "hobbies": "Hobbies and leisure activities",
    "lore": "Cultural topics and general knowledge",
    "belles_lettres": "Literary and essay writing",
    "government": "Government and public affairs",
    "learned": "Academic and scholarly writing",
    "fiction": "General fiction",
    "mystery": "Mystery fiction",
    "science_fiction": "Science-fiction stories",
    "adventure": "Adventure fiction",
    "romance": "Romance fiction",
    "humor": "Humorous writing"
}


def get_brown_categories():

    return list(brown.categories())


def format_category_inventory():

    categories = get_brown_categories()

    output = "Brown Corpus Categories:\n\n"

    for index, category in enumerate(categories, start=1):

        description = CATEGORY_DESCRIPTIONS.get(
            category,
            "Text category"
        )

        output += (
            f"{index}. {category.replace('_', ' ').title()}"
            f" — {description}\n"
        )

    output += (
        f"\nTotal categories: {len(categories)}"
    )

    return output


def is_category_inventory_request(text):

    text = normalize_text(text)

    patterns = [
        "brown corpus categories",
        "brown corpus genres",
        "categories in brown corpus",
        "genres in brown corpus",
        "what categories are available",
        "what genres are available",
        "list the categories",
        "list the genres",
        "show all categories",
        "show all genres",
        "all brown categories",
        "all brown genres",
        "brown corpus category list",
        "brown corpus genre list"
    ]

    return any(pattern in text for pattern in patterns)


# =========================================================
# 13. GENRE DETECTION
# =========================================================

def detect_genre(text):

    normalized = normalize_text(text)

    for category, aliases in GENRE_ALIASES.items():

        for alias in aliases:

            alias_normalized = normalize_text(alias)

            if re.search(
                r"\b" + re.escape(alias_normalized) + r"\b",
                normalized
            ):

                if category in brown.categories():
                    return category

    return None


# =========================================================
# 14. CATEGORY-SPECIFIC INDEX
# =========================================================

def build_genre_index(category):

    if category not in brown.categories():
        return None

    if category in genre_indexes:
        return genre_indexes[category]

    print("Building genre index:", category)

    sentences = []

    for sentence in brown.sents(categories=category):

        text = normalize_text(" ".join(sentence))

        word_count = len(text.split())

        if 6 <= word_count <= 40:
            sentences.append(text)

    if not sentences:
        return None

    category_vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=12000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    matrix = category_vectorizer.fit_transform(sentences)

    genre_indexes[category] = (
        sentences,
        category_vectorizer,
        matrix
    )

    return genre_indexes[category]


# =========================================================
# 15. RETRIEVAL ENGINE
# =========================================================

def retrieve_from_index(
    query,
    sentences,
    selected_vectorizer,
    selected_matrix,
    top_n=5,
    threshold=0.045
):

    if not sentences:
        return []

    query = normalize_text(query)

    query_vector = selected_vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        selected_matrix
    ).flatten()

    ranked_indices = similarities.argsort()[::-1]

    results = []
    seen = set()

    for index in ranked_indices:

        score = float(similarities[index])

        # Stop once scores become too weak
        if score < threshold:
            break

        sentence = sentences[index]

        if sentence in seen:
            continue

        seen.add(sentence)

        results.append({
            "sentence": sentence,
            "score": score
        })

        if len(results) >= top_n:
            break

    return results


def search_corpus(query, top_n=5, genre=None):

    # First search the requested genre
    if genre:

        genre_index = build_genre_index(genre)

        if genre_index is not None:

            sentences, genre_vectorizer, genre_matrix = genre_index

            results = retrieve_from_index(
                query,
                sentences,
                genre_vectorizer,
                genre_matrix,
                top_n=top_n,
                threshold=0.035
            )

            if results:
                return results

    # If genre search is weak, search the complete corpus
    build_corpus_index()

    return retrieve_from_index(
        query,
        corpus_sentences,
        vectorizer,
        corpus_matrix,
        top_n=top_n,
        threshold=0.045
    )


# =========================================================
# 16. SENTIMENT ANALYSIS
# =========================================================

POSITIVE_WORDS = {
    "happy", "joy", "good", "great", "excellent",
    "love", "wonderful", "amazing", "excited",
    "fantastic", "enjoy", "beautiful", "positive",
    "glad", "awesome", "success"
}

NEGATIVE_WORDS = {
    "sad", "bad", "terrible", "hate", "angry",
    "upset", "worried", "awful", "poor", "negative",
    "disappointed", "stress", "stressed", "fear",
    "hard", "difficult", "annoyed"
}


def detect_sentiment(tokens):

    positive_count = sum(
        1 for word in tokens if word in POSITIVE_WORDS
    )

    negative_count = sum(
        1 for word in tokens if word in NEGATIVE_WORDS
    )

    if positive_count > negative_count:
        return "Positive"

    elif negative_count > positive_count:
        return "Negative"

    return "Neutral"


# =========================================================
# 17. INTENT RECOGNITION
# =========================================================

def detect_intent(text, analysis):

    normalized = analysis["Normalized Text"]
    words = analysis["Lemmatization"]

    # Greeting
    if re.search(
        r"\b(hello|hi|hey|good morning|good evening)\b",
        normalized
    ):
        return "greeting"

    # Small talk
    if any(phrase in normalized for phrase in [
        "how are you", "how are u",
        "how do you feel", "what are you doing"
    ]):
        return "small_talk"

    # Identity
    if any(phrase in normalized for phrase in [
        "your name", "who are you", "about yourself"
    ]):
        return "identity"

    # Capabilities
    if any(phrase in normalized for phrase in [
        "what can you do", "your capabilities",
        "how can you help"
    ]):
        return "capabilities"

    # Project explanation
    if any(phrase in normalized for phrase in [
        "your project", "this project",
        "project details", "techniques used",
        "how you work", "how does this chatbot work"
    ]):
        return "project"

    # Thanks
    if any(word in words for word in [
        "thank", "thanks", "grateful"
    ]):
        return "thanks"

    # Goodbye
    if re.search(r"\b(bye|goodbye|exit|quit)\b", normalized):
        return "goodbye"

    # Help
    if "help" in words:
        return "help"

    # Brown Corpus category listing
    if is_category_inventory_request(text):
        return "category_inventory"

    # Sentiment detection only on statements
    if not normalized.startswith((
        "why ", "what ", "how ", "when ",
        "where ", "who ", "which "
    )):

        sentiment = detect_sentiment(words)

        if sentiment != "Neutral":
            return "sentiment"

    return "corpus_search"


# =========================================================
# 18. NLP CONCEPTS
# =========================================================

CONCEPTS = {
    "tokenization": (
        "Tokenization breaks text into smaller units such as "
        "words or sentences. These units are called tokens."
    ),

    "stemming": (
        "Stemming reduces words to a shorter root form by removing "
        "prefixes or suffixes."
    ),

    "lemmatization": (
        "Lemmatization converts a word into its meaningful dictionary "
        "base form using vocabulary and grammatical information."
    ),

    "stop words": (
        "Stop words are frequently occurring words such as the, is, "
        "and, and of that are often removed during text processing."
    ),

    "pos tagging": (
        "Part-of-speech tagging identifies grammatical categories "
        "of words such as nouns, verbs, adjectives and adverbs."
    ),

    "normalization": (
        "Text normalization converts text into a consistent format, "
        "such as lowercase letters and removing unnecessary characters."
    ),

    "tf idf": (
        "TF-IDF measures the importance of a word in a document "
        "relative to a collection of documents. It helps represent "
        "text numerically for information retrieval."
    ),

    "brown corpus": (
        "The Brown Corpus is a collection of English text samples "
        "from different genres. It is used for NLP research and "
        "language analysis."
    ),

    "sentiment analysis": (
        "Sentiment analysis identifies the emotional tone of text, "
        "such as positive, negative or neutral sentiment."
    ),

    "cosine similarity": (
        "Cosine similarity compares the angle between text vectors. "
        "This chatbot uses it to rank sentences by similarity to a query."
    )
}


def concept_response(normalized):

    for concept, answer in CONCEPTS.items():

        if concept in normalized:
            return answer

    normalized_compact = normalized.replace(" ", "")

    for concept, answer in CONCEPTS.items():

        if concept.replace(" ", "") in normalized_compact:
            return answer

    return None


# =========================================================
# 19. CHATBOT RESPONSE
# =========================================================

def chatbot_response(text, analysis, context=None):

    if context is None:
        context = {}

    normalized = analysis["Normalized Text"]
    words = analysis["Lemmatization"]

    intent = detect_intent(text, analysis)

    # Greeting
    if intent == "greeting":
        return (
            "Hello! I'm your NLP chatbot. Ask me a question "
            "or search the Brown Corpus."
        ), context

    # Small talk
    if intent == "small_talk":
        return (
            "I'm ready to process your text and help you "
            "explore language."
        ), context

    # Identity
    if intent == "identity":
        return (
            "I'm a Python chatbot built using NLTK. I use text "
            "preprocessing, WordNet, sentiment analysis and Brown "
            "Corpus retrieval."
        ), context

    # Capabilities
    if intent == "capabilities":
        return (
            "I can perform text normalization, tokenization, "
            "stopword removal, POS tagging, stemming, lemmatization, "
            "basic sentiment analysis, WordNet definitions and "
            "Brown Corpus sentence retrieval."
        ), context

    # Project information
    if intent == "project":
        return (
            "This chatbot uses Python and NLTK. Its NLP pipeline "
            "includes normalization, tokenization, stopword removal, "
            "POS tagging, stemming and lemmatization. It uses TF-IDF "
            "and cosine similarity for Brown Corpus retrieval, WordNet "
            "for definitions and a basic sentiment lexicon."
        ), context

    # Thanks
    if intent == "thanks":
        return "You're welcome!", context

    # Goodbye
    if intent == "goodbye":
        return "Goodbye! Thanks for chatting.", context

    # Help
    if intent == "help":
        return (
            "Try asking about NLP concepts, word meanings, sentiment, "
            "Brown Corpus genres, or retrieve sentences about a topic."
        ), context

    # Brown Corpus category list
    if intent == "category_inventory":
        return format_category_inventory(), context

    # NLP explanation
    if "nlp" in normalized.split() or "natural language processing" in normalized:
        return (
            "Natural Language Processing (NLP) is a branch of Artificial "
            "Intelligence that enables computers to process and analyze "
            "human language. It is used in chatbots, translation, "
            "sentiment analysis and text classification."
        ), context

    # Concepts
    concept = concept_response(normalized)

    if concept:
        return concept, context

    # WordNet definitions
    definition_match = re.match(
        r"^(what is|what are|define|meaning of|definition of)\s+(.+)$",
        normalized
    )

    if definition_match:

        search_word = definition_match.group(2).split()[0]

        definition = get_definition(search_word)

        if definition:
            return f"{search_word.title()} means: {definition}", context

    # Follow-up questions
    if any(phrase in normalized for phrase in [
        "tell me more", "more about it",
        "another example", "explain more"
    ]):

        query = context.get("last_query", text)

    else:
        query = text

    # Brown Corpus retrieval
    try:

        cleaned_query, genre = clean_retrieval_query(query)

        results = search_corpus(
            cleaned_query,
            top_n=5,
            genre=genre
        )

        if results:

            context["last_query"] = text

            answer = "Brown Corpus Retrieval"

            if genre:
                answer += f" | Genre: {genre.replace('_', ' ').title()}"

            answer += "\n\n"

            for i, result in enumerate(results, start=1):

                answer += (
                    f"{i}. {result['sentence'].capitalize()}\n"
                )

            answer += (
                "\nThese are original sentences from the Brown Corpus, "
                "ranked using TF-IDF and cosine similarity. They are "
                "retrieved text, not AI-generated answers."
            )

            return answer, context

    except Exception as error:

        print("Corpus retrieval error:", error)

    return (
        "I couldn't find sufficiently relevant sentences in the Brown "
        "Corpus. Try using different keywords or a broader topic."
    ), context