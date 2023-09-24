// Create a WebSocket connection
const ws = new WebSocket("ws://localhost:8000/clock");

// Function to update the clock
function updateClock(time) {
    document.getElementById("clock").innerText = time;
    if (time == "00:00") {
        fetchImage();
    }
}

// Listen for messages from the server
ws.addEventListener("message", function(event) {
    const data = JSON.parse(event.data);
    const time = data.time;
    updateClock(time);
});

// Initialize session when the page loads
window.onload = function() {
    // Check if session_id cookie is already set
    if (!document.cookie.includes("session_id")) {
        fetch("/init/")
        .then(response => response.json())
        .then(data => {
            // Session initialized and cookie set by the server
            console.log("Session initialized:", data.session_id);
        })
        .catch(error => console.error("Error:", error));
    } else {
        console.log("Session already initialized");
    }
};

const imageElement = document.getElementById("generated-image");
const submitButton = document.getElementById("submit-button");
const userInput = document.getElementById("user-input");

// Fetch initial image
fetchImage();

// Event listener for the submit button
submitButton.addEventListener("click", function() {
    const guess1 = document.getElementById("user-input1").value;
    const guess2 = document.getElementById("user-input2").value;

    if (guess1 && guess2) {
        fetch("/compute_score/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ guess1: guess1, guess2: guess2 }),
        })
        .then(response => response.json())
        .then(data => {
            const score1 = data.score1; // assuming the backend sends score1
            const score2 = data.score2; // assuming the backend sends score2

            // Update the UI with the new score
            const scoreElement1 = document.getElementById("score1");
            const scoreElement2 = document.getElementById("score2");

            scoreElement1.textContent = `Score: ${score1}`;
            scoreElement2.textContent = `Score: ${score2}`;

            scoreElement1.style.display = "block";
            scoreElement2.style.display = "block";

            // Fetch and display the new image
            fetchImage();
        })
        .catch(error => console.error("Error:", error));
    }
});

// Function to fetch and display image
function fetchImage() {
    fetch("/fetch_image/")
    .then(response => response.blob())
    .then(blob => {
    const imageUrl = URL.createObjectURL(blob);
    imageElement.src = imageUrl;
    })
    .catch(error => console.error("Error fetching image:", error));
}