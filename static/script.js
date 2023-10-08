let dictionary = null;

fetch('./dict/en_US.aff').then(response => response.text()).then((affData) => {
    fetch('./dict/en_US.dic').then(response => response.text()).then((dicData) => {
        dictionary = new Typo("en_US", affData, dicData);
        console.log("Dictionary Loaded")
    });
});

// document.addEventListener("DOMContentLoaded", () => {
//     createClockElement();
//     initializeApp().then(app => {
//         // Use app to get the session ID and refresh content
//         createSubmitButton(app); // Pass app to createSubmitButton
//         app.initializeSession()
//             .then(() => {
//                 app.initializeWebSocket();
//                 app.fetchAndDisplayContents(true);
//             })
//             .catch(err => console.error("Error initializing session:", err));
//     });
// });

document.addEventListener("DOMContentLoaded", () => {
    createClockElement();
    initializeApp().then(app => {
        // Use app to get the session ID and refresh content
        createSubmitButton(app); // Pass app to createSubmitButton

        app.initializeSession().then(() => {
            app.initializeWebSocket();

            // Check the client's status after initializing the session
            fetch("/client/status", {
                method: "GET",
                credentials: 'include', // to ensure cookies are sent with the request
            })
            .then(response => response.json())
            .then(data => {
                if (data.hasWon === 1) {
                    app.fetchAndDisplayContents(false);
                } else {
                    app.fetchAndDisplayContents(true);
                }
            })
            .catch(error => {
                console.error("Error checking client status:", error);
            });
        })
        .catch(err => console.error("Error initializing session:", err));
    });
});

function clearPrompt() {
    const promptContainer = document.getElementById("prompt-container");
    while (promptContainer.firstChild) {
        promptContainer.firstChild.remove();
    }
}

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
                clearPrompt();
                fetchAndDisplayContents(true);
            }
        });
    }

    async function fetchAndDisplayContents(prompt) {
        try {
            const response = await fetch(`/fetch/contents?session_id=${sessionId}`);
            const data = await response.json();
            displayImage(data.image);
            if (prompt === true) {
                displayPrompt(data.prompt);
            } else {
                displayPrompt({
                    tokens: ['Congratulations, you got it!', 'Good luck next round.'],
                    masks: [],
                });
            }
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
    clockDiv.style.backgroundColor = 'rgba(255, 255, 255, 0.4)';
    clockDiv.style.padding = '10px 20px';
    clockDiv.style.borderRadius = '15px';
    clockDiv.style.boxShadow = '0px 0px 15px 5px rgba(255, 255, 255, 0.9)';
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
    submitButton.style.marginTop = '3px';
    submitButton.style.marginBottom = '3px';
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
        this.style.backgroundColor = "rgba(180, 180, 180, 0.7)";
    };

    submitButton.onmouseout = function() {
        this.style.backgroundColor = "transparent"; // Green
    };

    // Add click event listener
    submitButton.addEventListener('click', () => submitInputs(app));
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            submitButton.click();
        }
    });
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
            inputField.style.fontSize = "20px";
            inputField.style.fontFamily = "Arial, sans-serif";
            inputField.style.margin = "3px";
            inputField.style.transition = 'background 0.3s';
            promptContainer.appendChild(inputField);
        } else {
            // Otherwise, add a span with the token
            const span = document.createElement("span");
            span.textContent = token + " ";
            span.style.fontSize = "18px";
            span.style.fontFamily = "Courier New, monospace";
            span.style.margin = "3px";
            promptContainer.appendChild(span);
        }
    });

    // Ensure the submit button is appended last
    const submitButton = document.getElementById('submit-button');
    if (submitButton) {
        if (masks.length > 0) {
            // If there are masks, ensure the button is attached and visible
            promptContainer.appendChild(submitButton);
        } else {
            // Otherwise, hide the button
            submitButton.style.display = 'none';
        }
    }
}

function flashRed(input) {
    input.style.background = 'red';
    setTimeout(() => {
        input.style.background = 'black';
    }, 150);
}

function submitInputs(app) {
    const inputs = [];
    const promptContainer = document.getElementById("prompt-container");
    
    // Get all input fields
    const inputFields = promptContainer.querySelectorAll("input");
    
    let hasAnyTypos = false;
    
    inputFields.forEach(input => {
        if (hasTypo(input.value)) {
            // Flash the input red if it has a typo
            flashRed(input);
            hasAnyTypos = true;
        } else {
            inputs.push(input.value);
        }
    });

    // If any input has a typo, don't send the data to the server
    if (hasAnyTypos) return;

    fetch("/compute_score", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ inputs })
    })
    .then(response => response.json())
    .then(data => {
        if (parseFloat(data.max) > 0.99) {
            app.fetchAndDisplayContents(false);
        } else {
            app.fetchAndDisplayContents(true);
        }
    })
    .catch(error => {
        console.error("Error submitting data:", error);
    });
}

function hasTypo(inputValue) {
    const words = inputValue.split(/\s+/);
    for (const word of words) {
        if (dictionary && !dictionary.check(word)) {
            return true;
        }
    }
    return false;
}