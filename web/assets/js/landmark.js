let hands;
let showLandmark = true;

function initMediaPipe(){

    hands = new Hands({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
        }
    });

    hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7
});

    hands.onResults(onHandResults);

    document.getElementById("landmarkStatus").innerText = "Aktif";
}

function onHandResults(results){

    const canvas = document.getElementById("landmarkCanvas");
    const ctx = canvas.getContext("2d");
    const video = document.getElementById("video");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.clearRect(0,0,canvas.width,canvas.height);

    if(!showLandmark) return;

    if(results.multiHandLandmarks){

        for(const landmarks of results.multiHandLandmarks){

            drawConnectors(
                ctx,
                landmarks,
                HAND_CONNECTIONS,
                {
                    color:"#00FF00",
                    lineWidth:3
                }
            );

            drawLandmarks(
                ctx,
                landmarks,
                {
                    color:"#FF0000",
                    radius:4
                }
            );
        }
    }
}