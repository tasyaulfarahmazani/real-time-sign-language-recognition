let currentMode = "letter";

let sentence = [];
let totalDetection = 0;
let acceptedDetection = 0;

let lastPrediction = "";

function switchMode(mode){

    currentMode = mode;

    document
        .querySelectorAll(".tab")
        .forEach(btn => btn.classList.remove("active"));

    document
        .querySelector(`[data-mode="${mode}"]`)
        .classList.add("active");

    document.getElementById("modeLabel")
        .innerText = mode;

}

function resetSentence(){

    sentence = [];

    totalDetection = 0;
    acceptedDetection = 0;

    lastPrediction = "";

    document.getElementById("statTotal").innerText = "0";
    document.getElementById("statAccepted").innerText = "0";

    renderSentence();

    showToast("Kalimat direset");
}

function copySentence(){

    const text =
        document.getElementById("sentenceTokens")
        .innerText;

    navigator.clipboard.writeText(text);

    showToast("Teks berhasil disalin");
}

function showToast(text){

    const toast = document.getElementById("toast");

    toast.innerText = text;
    toast.classList.add("show");

    setTimeout(()=>{
        toast.classList.remove("show");
    },2000);
}

function setPrediction(label, confidence){

    document.getElementById("predValue")
        .innerText = label || "—";

    document.getElementById("confFill")
        .style.width = confidence + "%";

    document.getElementById("confPct")
        .innerText = confidence + "%";

    // hitung total deteksi
    totalDetection++;

    document.getElementById("statTotal")
        .innerText = totalDetection;

    // threshold 40%
    if(label && confidence >= 40){

        acceptedDetection++;

        document.getElementById("statAccepted")
            .innerText = acceptedDetection;

        // hindari duplikat terus menerus
        if(label !== lastPrediction){

            sentence.push(label);

            renderSentence();

            lastPrediction = label;
        }
    }
}

function renderSentence(){

    const container =
        document.getElementById("sentenceTokens");

    if(sentence.length === 0){

        container.innerHTML =
        `<span class="token-placeholder">
            Mulai deteksi untuk melihat hasil...
        </span>`;

        return;
    }

    container.innerHTML = sentence
        .map(word =>
            `<span class="token">${word}</span>`
        )
        .join("");
}