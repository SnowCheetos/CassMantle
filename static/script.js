document.addEventListener("DOMContentLoaded", () => {
    createClockElement();
    initializeApp().then(app => {
        // Use app to get the session ID and refresh content
        createSubmitButton(app); // Pass app to createSubmitButton
        app.initializeSession()
            .then(() => {
                app.initializeWebSocket();
                app.fetchAndDisplayContents();
            })
            .catch(err => console.error("Error initializing session:", err));
    });
});

function initializeApp() {
    let sessionId;

    async function initializeSession() {
        try {
            sessionId = localStorage.getItem('session_id');
            if (!sessionId) {
                const response = await fetch("/init");
                const data = await response.json();
                console.log("Session initialized:", data.session_id);
                sessionId = data.session_id;
    
                // Store session_id in localStorage
                localStorage.setItem('session_id', sessionId);
            } else {
                console.log("Session ID retrieved from localStorage:", sessionId);
            }
            return sessionId;
        } catch (err) {
            throw err;
        }
    }    

    function initializeWebSocket() {
        const ws = new WebSocket("ws://localhost:8000/clock");

        ws.addEventListener("message", event => {
            const data = JSON.parse(event.data);
            updateClock(data.time);

            if (data.reset) {
                fetchAndDisplayContents();
            }
        });
    }

    async function fetchAndDisplayContents() {
        try {
            const response = await fetch(`/fetch/contents?session_id=${sessionId}`);
            const data = await response.json();
            displayImage(data.image);
            displayPrompt(data.prompt);
        } catch (err) {
            console.error("Error fetching contents:", err);
        }
    }

    function getSessionId() {
        return sessionId;
    }

    return Promise.resolve({
        initializeSession,
        initializeWebSocket,
        fetchAndDisplayContents,
        getSessionId
    });
}

function updateClock(time) {
    // Update the DOM with the new time
    const clockElement = document.getElementById('clock');
    clockElement.textContent = time;
}

function displayImage(imageData) {
    const imageElement = document.getElementById("generated-image");
    imageElement.src = `data:image/jpeg;base64,${imageData}`;
}

function createClockElement() {
    // Get the app container
    const appContainer = document.getElementById('app');

    // Create the clock div
    const clockDiv = document.createElement('div');
    clockDiv.id = 'clock';
    clockDiv.textContent = '00:00';

    // Append the clock div to the app container
    appContainer.appendChild(clockDiv);

    // Add CSS styles dynamically
    clockDiv.style.fontFamily = 'Orbitron, sans-serif';
    clockDiv.style.position = 'fixed';
    clockDiv.style.top = '10px';
    clockDiv.style.left = '50%';
    clockDiv.style.transform = 'translateX(-50%)';
    clockDiv.style.fontSize = '2em';
    clockDiv.style.textAlign = 'center';
    clockDiv.style.textShadow = '2px 2px 4px rgba(255, 255, 255, 0.5)';
}

function createSubmitButton(app) {
    // Get the prompt container
    const promptContainer = document.getElementById('prompt-container');

    // Create the submit button
    const submitButton = document.createElement('button');
    submitButton.id = 'submit-button';
    submitButton.textContent = 'SUBMIT';

    // Append the submit button to the prompt container
    promptContainer.appendChild(submitButton);

    // Add CSS styles dynamically
    submitButton.style.fontSize = '16';
    submitButton.style.marginLeft = '15px';
    submitButton.style.marginTop = '2px';
    submitButton.style.marginBottom = '2px';
    submitButton.style.padding = '4px';
    submitButton.style.paddingLeft = '15px';
    submitButton.style.paddingRight = '15px';
    submitButton.style.backgroundColor = 'transparent'; // Green
    submitButton.style.color = 'white';
    submitButton.style.border = '2px solid white';
    submitButton.style.borderRadius = '20px';
    submitButton.style.cursor = 'pointer';
    submitButton.style.fontFamily = "Tahoma, sans-serif";

    // Add hover effect
    submitButton.onmouseover = function() {
        this.style.backgroundColor = "rgba(180, 180, 180, 0.7)"; //"#45a049"; // Darker green
    };

    submitButton.onmouseout = function() {
        this.style.backgroundColor = "transparent"; // Green
    };

    // Add click event listener
    submitButton.addEventListener('click', () => submitInputs(app));
}

function displayPrompt(promptData) {
    const { tokens, masks } = promptData;
    const promptContainer = document.getElementById("prompt-container");
    
    // Clear any existing content, but keep the submit button
    while (promptContainer.firstChild && promptContainer.firstChild.id !== 'submit-button') {
        promptContainer.firstChild.remove();
    }
    
    tokens.forEach((token, index) => {
        if (masks.includes(index)) {
            // If the index is in masks, add an input field
            const inputField = document.createElement("input");
            inputField.type = "text";
            inputField.id = `input-${index}`;
            inputField.style.border = 'none';
            inputField.style.backgroundColor = 'black';
            inputField.style.color = 'white';
            inputField.style.fontSize = "20px";  // Add inline style for font size
            inputField.style.fontFamily = "Arial, sans-serif";  // Add inline style for font family
            inputField.style.margin = "1px";
            promptContainer.appendChild(inputField);
        } else {
            // Otherwise, add a span with the token
            const span = document.createElement("span");
            span.textContent = token + " ";  // Add a space for separation
            span.style.fontSize = "18px";  // Add inline style for font size
            span.style.fontFamily = "Courier New, monospace";  // Add inline style for font family
            span.style.margin = "1px";
            promptContainer.appendChild(span);
        }
    });

    // Ensure the submit button is appended last
    const submitButton = document.getElementById('submit-button');
    if (submitButton) {
        promptContainer.appendChild(submitButton);
    }
}

function submitInputs(app) {
    // Logic for gathering user inputs and sending them to the server
    const inputs = [];
    const promptContainer = document.getElementById("prompt-container");
    
    // Get all input fields
    const inputFields = promptContainer.querySelectorAll("input");
    inputFields.forEach(input => {
        inputs.push(input.value);
    });

    fetch("/compute_score", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ inputs })
    })
    .then(response => response.json())
    .then(data => {
        app.fetchAndDisplayContents();
    })
    .catch(error => {
        console.error("Error submitting data:", error);
    });
}