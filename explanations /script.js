// Array of possible goals
const goals = ["Dots", "EnergyPill", "Survival", "LoseALife"];
const explanationTypes = [
    "dataset-similarity-explanation",
    "perturbation-based-saliency-map",
    "trd-natural-language",
    "optimal-action",
    "ganterfactual",
];

let currentGoal = '';

// Function to generate a new set of observation and explanation
function generateNewSet() {
    const obsNum = Math.floor(Math.random() * 60);
    const goalIndex = Math.floor(Math.random() * 4);
    const explanationIndex = Math.floor(Math.random() * 3);
    currentGoal = goals[goalIndex];
    const currentExplanation = explanationTypes[explanationIndex];

    console.log(`obsNum=${obsNum}, goalIndex=${goalIndex}, explanationIndex=${explanationIndex}`);
    console.log(`obs: ${obsNum}, goal: ${currentGoal}, explanation: ${currentExplanation}`);

    document.getElementById("observation-image").src = `explanation-obs/obs-${obsNum}.png`;

    if (currentExplanation === "dataset-similarity-explanation") {
        document.getElementById("explanation-image").style.display = "none";
        document.getElementById("explanation-video").style.display = "block";
        document.getElementById("explanation-video-source").src = `${currentExplanation}/dse-obs-${obsNum}-agent-${currentGoal}-v2.mp4`;
        document.getElementById("explanation-video").load(); // Reload video element
        console.log("loaded dse");
    } else if (currentExplanation === "perturbation-based-saliency-map") {
        document.getElementById("explanation-video").style.display = "none";
        document.getElementById("explanation-image").style.display = "block";
        document.getElementById("explanation-image").src = `${currentExplanation}/pbsm-obs-${obsNum}-agent-${currentGoal}-v2.png`;
        console.log("loaded pbsm");
    } else if (currentExplanation === "trd-natural-language") {
        document.getElementById("explanation-video").style.display = "none";
        document.getElementById("explanation-image").style.display = "block";
        document.getElementById("explanation-image").src = `${currentExplanation}/trd-obs-${obsNum}-agent-${currentGoal}-v2.png`;
        console.log("loaded trd");
    } else {
        console.log(`unknown ${currentExplanation}`);
    }

    // Clear previous result and radio button selection
    // document.getElementById("result").innerText = '';
    const goal_radios = document.querySelectorAll('input[name="goal"]');
    goal_radios.forEach(radio => radio.checked = false);
    const confidence_radios = document.querySelectorAll('input[name="confidence"]')
    confidence_radios.forEach(radio => radio.checked = false);
}

// Initial call to load the first set
generateNewSet();

// Handle the form submission
document.getElementById("submit-btn").addEventListener("click", function () {
    const selectedGoal = document.querySelector('input[name="goal"]:checked');

    if (selectedGoal) {
        const resultText = selectedGoal.value === currentGoal ? "Correct!" : `Wrong! The correct goal was ${currentGoal}.`;
        document.getElementById("result").innerText = resultText;
        generateNewSet();
    } else {
        document.getElementById("result").innerText = "Please select a goal before submitting.";
    }
});

// Handle generating a new set of observation and explanation
document.getElementById("new-set-btn").addEventListener("click", generateNewSet);
