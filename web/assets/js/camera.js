let fpsCounter = 0;
let fpsTime = performance.now();
let stream = null;
let cameraRunning = false;

async function startCamera(){

    const video = document.getElementById("video");
    const idleOverlay = document.getElementById("idleOverlay");

    try{

        stream = await navigator.mediaDevices.getUserMedia({
            video: true
        });

        video.srcObject = stream;
        video.classList.add("active");


        idleOverlay.classList.add("hidden");

        cameraRunning = true;
        landmarkLoop();

    }catch(err){

        alert("Tidak dapat mengakses kamera");
        console.log(err);

    }
document.getElementById("btnStart").innerHTML = `
<svg viewBox="0 0 24 24" fill="currentColor">
    <rect x="6" y="6" width="12" height="12"/>
</svg>
Hentikan Kamera
`;

}

function stopCamera(){

    if(stream){

        stream.getTracks().forEach(track => track.stop());

        stream = null;
    }

    document.getElementById("video")
        .classList.remove("active");

    document.getElementById("idleOverlay")
        .classList.remove("hidden");

    cameraRunning = false;
    setPrediction("",0);

document.getElementById("btnStart").innerHTML = `
<svg viewBox="0 0 24 24" fill="currentColor">
    <path d="M8 5v14l11-7z"/>
</svg>
Mulai Kamera
`;
}

function toggleCamera(){

    if(cameraRunning){
        stopCamera();
    }else{
        startCamera();
    }

}

async function sendFrame(){

    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");

    // sesuaikan ukuran canvas dengan video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");

    // kalau video masih mirror pakai scaleX(-1)
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
    ctx.restore();

    canvas.toBlob(async (blob) => {

        try {
            console.log("Mode aktif:", currentMode);
            const result = await sendPrediction(blob, currentMode);

            console.log("HASIL API:", result);

            setPrediction(
                result.prediction,
                result.confidence
            );

        } catch(err) {

            console.log(err);

        }

    }, "image/jpeg");
}

let frameCounter = 0;

async function landmarkLoop(){

    if(!cameraRunning) return;

    const video = document.getElementById("video");

    if(video.readyState >= 2){

        await hands.send({image: video});

        frameCounter++;

        if(frameCounter % 15 === 0){
            sendFrame();
        }
    }

    requestAnimationFrame(landmarkLoop);
}