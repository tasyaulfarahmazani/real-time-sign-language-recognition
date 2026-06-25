window.addEventListener("load", () => {

    checkAPI();

    setInterval(
        checkAPI,
        8000
    );

    initMediaPipe();

});