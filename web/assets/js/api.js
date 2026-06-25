const API_BASE = "http://127.0.0.1:8000";

const SESSION_ID = crypto.randomUUID();

async function checkAPI() {

    const apiDot = document.getElementById("apiDot");
    const apiStatus = document.getElementById("apiStatus");

    try {

        const response = await fetch("http://127.0.0.1:8000/docs");

        if(response.ok){

            apiDot.classList.add("online");
            apiStatus.textContent = "Terhubung";

        }else{

            apiDot.classList.add("offline");
            apiStatus.textContent = "Backend Offline";
        }

    } catch (error) {

        apiDot.classList.add("offline");
        apiStatus.textContent = "Tidak Terhubung";

        console.error(error);
    }
}

async function sendPrediction(blob, mode){

    const endpoint =
        mode === "word"
        ? "/predict-dynamic"
        : "/predict";

    const formData = new FormData();
    formData.append("file", blob);

    let url = API_BASE + endpoint;

    
    if(mode !== "word"){
        url += `?mode=${mode}`;
    }

    if(mode === "word"){
        url += `?session_id=${SESSION_ID}`;
    }

    const response = await fetch(url,{
        method:"POST",
        body:formData
    });

    return await response.json();
}