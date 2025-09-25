
// Helper functions for UI
function setStatus(text) {
	$('#status-text').html(text);
}

function setProgress(value) {
	const percentage = Math.round(value * 100);
	$('#progress-bar').css('width', percentage + '%');
	$('#progress-container').show();
	setStatus(`กำลังโหลดโมเดล... ${percentage}%`);
}

// Define 2 helper functions
function simulateClick(tabID) {
	document.getElementById(tabID).click();
}

function predictOnLoad() {
	// Simulate a click on the predict button
	setTimeout(simulateClick.bind(null,'predict-button'), 500);
}

// LOAD THE MODEL
let model;
(async function () {
	setStatus('กำลังโหลดโมเดล...');
	try {
		model = await tf.loadModel(
			// This model path is a placeholder. Please update with the correct path.
			'https://muaxzinn.github.io/SBLT-CAS-Skin-Brain-Lung-Eye-Classification-and-Segmentation/Diabetic_Retinopathy/model_dr_2/model.json',
			{ onProgress: setProgress }
		);
		// เมื่อโหลดเสร็จ ให้ซ่อน Progress Bar และแสดงสถานะพร้อมใช้งาน
		$('#progress-container').hide();
		setStatus('โมเดลพร้อมใช้งาน ✅');
		// The default image is already set in HTML, so we just trigger prediction.
		predictOnLoad();
	} catch (e) {
		setStatus('โหลดโมเดลล้มเหลว ❌');
		console.error(e);
	}
})();

$("#predict-button").click(async function () {
	let image = $('#selected-image').get(0);
	
	// Pre-process the image
	let tensor = tf.browser.fromPixels(image)
		.resizeNearestNeighbor([224, 224]) // Adjust size if needed
		.toFloat()
		.sub(127.5)
		.div(127.5)
		.expandDims();

	let predictions = await model.predict(tensor).data();
	let top5 = Array.from(predictions)
		.map(function (p, i) {
			return {
				probability: p,
				className: TARGET_CLASSES[i] 
			};
		}).sort(function (a, b) {
			return b.probability - a.probability;
		}).slice(0, 5); // Show top 5 results

	var file_name = 'ภาพตัวอย่าง.jpg';
	
	// Clear previous results
	$("#prediction-results-container").empty();

	// Append the file name to the prediction list
	$("#prediction-results-container").append(`<div class="w3-text-blue fname-font w3-margin-bottom"><b>${file_name}</b></div>`);
	
	top5.forEach(function (p) {
		const percentage = (p.probability * 100).toFixed(1);
		$("#prediction-results-container").append(`
			<div class="prediction-item w3-card-2 w3-round-large w3-margin-bottom w3-padding-small">
				<div class="w3-row w3-small">
					<div class="w3-col s8 m9 l9">
						<span class="prediction-label w3-text-dark-grey">${p.className}</span>
					</div>
					<div class="w3-col s4 m3 l3 w3-right-align">
						<span class="prediction-probability w3-text-blue"><b>${percentage}%</b></span>
					</div>
				</div>
				<div class="prediction-bar-container w3-light-grey w3-round-large" style="height: 8px;">
					<div class="prediction-bar w3-blue w3-round-large" style="width: ${percentage}%"></div>
				</div>
			</div>
		`);
	});
});

$("#image-selector").change(async function () {
	fileList = $("#image-selector").prop('files');
	model_processArray(fileList); // This function is in app_batch_prediction_code.js
});
