import numpy as np


SUITABLE_LABEL = "Suitable for Children"
NOT_SUITABLE_LABEL = "Not Suitable for Children"
GENERIC_REVIEW_TERMS = {
    "character",
    "characters",
    "different",
    "film",
    "lot",
    "lots",
    "many",
    "movie",
    "part",
    "parts",
    "plot",
    "review",
    "scene",
    "scenes",
    "setting",
    "settings",
    "story",
}


def get_linear_coefficients(model):
    if hasattr(model, "coef_"):
        return model.coef_[0]

    if not hasattr(model, "calibrated_classifiers_"):
        return None

    coefficients = []

    for calibrated_model in model.calibrated_classifiers_:
        estimator = getattr(calibrated_model, "estimator", None)
        if estimator is None:
            estimator = getattr(calibrated_model, "base_estimator", None)

        if hasattr(estimator, "coef_"):
            coefficients.append(estimator.coef_[0])

    if not coefficients:
        return None

    return np.mean(coefficients, axis=0)


def get_feature_names(vectorizer):
    if not hasattr(vectorizer, "get_feature_names_out"):
        return None

    return vectorizer.get_feature_names_out()


def clean_feature_name(feature_name):
    if "__" in feature_name:
        return feature_name.split("__", 1)[1]

    return feature_name


def is_explainable_feature(feature_name):
    return not feature_name.startswith("char__")


def is_generic_review_term(term):
    tokens = term.split()

    return all(token in GENERIC_REVIEW_TERMS for token in tokens)


def is_redundant_term(term, selected_terms):
    term_tokens = set(term.split())

    for selected_term in selected_terms:
        selected_tokens = set(selected_term.split())

        if term == selected_term:
            return True

        if term_tokens.issubset(selected_tokens):
            return True

        if selected_tokens.issubset(term_tokens):
            return True

    return False


def get_supporting_terms(review_vector, vectorizer, model, prediction, top_n=2):
    coefficients = get_linear_coefficients(model)
    feature_names = get_feature_names(vectorizer)

    if coefficients is None or feature_names is None:
        return []

    weighted_features = review_vector.multiply(coefficients).tocoo()
    supporting_terms = []

    for column_index, contribution in zip(
        weighted_features.col, weighted_features.data
    ):
        feature_name = feature_names[column_index]
        term = clean_feature_name(feature_name)

        if not is_explainable_feature(feature_name):
            continue

        if is_generic_review_term(term):
            continue

        if prediction == 1 and contribution > 0:
            supporting_terms.append((term, contribution))
        elif prediction == 0 and contribution < 0:
            supporting_terms.append((term, -contribution))

    supporting_terms.sort(
        key=lambda item: (item[1], len(item[0].split())), reverse=True
    )

    unique_terms = []
    seen_terms = set()

    for term, _ in supporting_terms:
        if term in seen_terms or is_redundant_term(term, unique_terms):
            continue

        unique_terms.append(term)
        seen_terms.add(term)

        if len(unique_terms) == top_n:
            break

    return unique_terms


def format_terms(terms):
    quoted_terms = [f'"{term}"' for term in terms]

    if len(quoted_terms) == 1:
        return quoted_terms[0]

    if len(quoted_terms) == 2:
        return " and ".join(quoted_terms)

    return ", ".join(quoted_terms[:-1]) + f", and {quoted_terms[-1]}"


def generate_model_reason(review_vector, vectorizer, model, prediction):
    terms = get_supporting_terms(review_vector, vectorizer, model, prediction)

    if prediction == 0:
        if terms:
            return (
                "The model weighted phrases such as "
                f"{format_terms(terms)} toward not suitable for children."
            )

        return (
            "The review does not provide enough child-suitability detail, "
            "so the conservative model did not mark it suitable."
        )

    if terms:
        return (
            "The model weighted phrases such as "
            f"{format_terms(terms)} toward suitable for children."
        )

    return "The model's learned patterns lean toward generally appropriate content."
