// Code is based on a YouTube tutorial by deeplizard
// https://www.youtube.com/watch?v=HEQDRWMK6yY

// Helper functions for UI
function setStatus(text) {
	// In your HTML, the progress bar div also serves as the status text
	$('.progress-bar').text(text);
}

function setProgress(value) {
	const percentage = Math.round(value * 100);
	// You might need a separate progress bar element if you want a visual bar.
	// For now, we'll just update the text.
	setStatus(`กำลังโหลดโมเดล... ${percentage}%`);
}


// After the model loads we want to make a prediction on the default image.
// Thus, the user will see predictions when the page is first loaded.

function simulateClick(tabID) {
	
	document.getElementById(tabID).click();
}

function predictOnLoad() {
	
	// Simulate a click on the predict button
	setTimeout(simulateClick.bind(null,'predict-button'), 500);
};


$("#image-selector").change(function () {
	let reader = new FileReader();
	reader.onload = function () {
		let dataURL = reader.result;
		$("#selected-image").attr("src", dataURL);
		$("#prediction-list").empty();
	}
	
		
		let file = $("#image-selector").prop('files')[0];
		reader.readAsDataURL(file);
		
		
		// Simulate a click on the predict button
		// This introduces a 0.5 second delay before the click.
		// Without this long delay the model loads but may not automatically
		// predict.
		setTimeout(simulateClick.bind(null,'predict-button'), 500);

});




let model;
(async function () {
	setStatus('กำลังโหลดโมเดล...');
	try {
		model = await tf.loadModel(
			'http://concept.test.woza.work/final_model_kaggle_version1/model.json',
			{ onProgress: setProgress } // This will call setProgress during loading
		);
		$("#selected-image").attr("src", "http://concept.test.woza.work/assets/samplepic.jpg");
		$('.progress-bar').hide(); // Hide loading text/bar on success
		// Simulate a click on the predict button for the default image
		predictOnLoad();
	} catch (e) {
		setStatus('โหลดโมเดลล้มเหลว');
		console.error(e);
		// Keep the error message visible
		$('.progress-bar').css('color', 'red');
	}
})();




$("#predict-button").click(async function () {
	
	
	
	let image = $('#selected-image').get(0);
	
	// Pre-process the image
	let tensor = tf.fromPixels(image)
	.resizeNearestNeighbor([224,224])
	.toFloat();
	
	
	let offset = tf.scalar(127.5);
	
	tensor = tensor.sub(offset)
	.div(offset)
	.expandDims();
	
	
	
	
	// Pass the tensor to the model and call predict on it.
	// Predict returns a tensor.
	// data() loads the values of the output tensor and returns
	// a promise of a typed array when the computation is complete.
	// Notice the await and async keywords are used together.
	let predictions = await model.predict(tensor).data();
	let top5 = Array.from(predictions)
		.map(function (p, i) { // this is Array.map
			return {
				probability: p,
				className: SKIN_CLASSES[i] // we are selecting the value from the obj
			};
				
			
		}).sort(function (a, b) {
			return b.probability - a.probability;
				
		}).slice(0, 7);
	
	
$("#prediction-list").empty();
top5.forEach(function (p) {

	$("#prediction-list").append(`<li>${p.className}: ${p.probability.toFixed(6)}</li>`);

	
	});
	
	
});
