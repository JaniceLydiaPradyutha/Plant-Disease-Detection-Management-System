let model;

// Load the TensorFlow.js model
async function loadModel() {
const status = document.getElementById("status");
try {
status.innerText = "🔄 Loading model...";
model = await tf.loadLayersModel("tensorflowjs-model/model.json");
status.innerText = "✅ Model loaded successfully!";
console.log("Model loaded:", model);
} catch (err) {
console.error("Model load error:", err);
status.innerText = "❌ Failed to load model.";
}
}

// Run prediction on an image element
async function predictDisease(imgElement) {
if (!model) {
return "Model not loaded.";
}

try {
// Preprocess the image
const tensor = tf.browser.fromPixels(imgElement)
.resizeNearestNeighbor([150, 150])
.toFloat()
.div(255.0)
.expandDims();

javascript
Copy
Edit
// Run prediction
const prediction = await model.predict(tensor).data();

// Class labels
const classes = ["Healthy", "Rust", "Powdery"];
const maxIndex = prediction.indexOf(Math.max(...prediction));
return classes[maxIndex];
} catch (error) {
console.error("Prediction error:", error);
return "Prediction failed.";
}
}

// Load model on DOM ready
window.addEventListener("DOMContentLoaded", loadModel);
