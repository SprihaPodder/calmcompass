import pickle
from sklearn.tree import DecisionTreeClassifier

# Sample training data (age, travel_freq (0=daily, 1=weekly, 2=rare), crowd_tolerance, noise_sensitivity, light_sensitivity)
X = [
    [19, 0, 5, 6, 4],
    [25, 1, 3, 3, 2],
    [30, 0, 8, 9, 7],
    [22, 2, 2, 2, 3],
    [40, 0, 9, 8, 9],
]

# Corresponding labels (simplified strategies & tools indexes)
y = [0, 1, 2, 1, 0]

# Mapping from label index to outputs
label_map = {
    0: {
        "optimal_times": "Early mornings and evenings",
        "tools": ["Noise-canceling headphones", "Soothing playlists", "Travel journal"],
        "strategies": ["Use well-lit paths", "Wear sunglasses", "Take breaks during long travel"]
    },
    1: {
        "optimal_times": "Midday when crowd is low",
        "tools": ["Breathing exercises", "Aromatherapy", "Calm app"],
        "strategies": ["Avoid peak hour", "Travel with a companion", "Use scenic routes"]
    },
    2: {
        "optimal_times": "Late evening, quieter periods",
        "tools": ["Fidget toys", "White noise machine", "Grounding techniques"],
        "strategies": ["Avoid crowded public transport", "Use personal vehicle", "Plan short trips"]
    }
}

# Train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model and label_map
with open("travel_model.pkl", "wb") as f:
    pickle.dump((model, label_map), f)