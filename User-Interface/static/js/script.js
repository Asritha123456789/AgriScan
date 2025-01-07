function updateFileName() {
    var input = document.getElementById('file');
    var fileName = input.files[0].name;
    document.getElementById('file-name').textContent = fileName;
}

function checkFileType() {
    const fileInput = document.getElementById("file");
    const filePath = fileInput.value;
    const allowedExtensions = /(\.jpg|\.jpeg|\.png)$/i;
    if (!allowedExtensions.exec(filePath)) {
        document.getElementById("error-message").style.display = "block";
        fileInput.value = '';
        return false;
    } else {
        document.getElementById("error-message").style.display = "none";
        return true;
    }
}

function clearImage() {
    document.getElementById('file').value = "";  // Clear file input
    document.getElementById('file-name').innerText = "No file chosen";
    const imagePreview = document.getElementById('image-preview');
    const predictionResult = document.getElementById('prediction-result');
    
    if (imagePreview) imagePreview.innerHTML = "";  // Clear the image preview
    if (predictionResult) predictionResult.innerHTML = "";  // Clear the prediction result
}
