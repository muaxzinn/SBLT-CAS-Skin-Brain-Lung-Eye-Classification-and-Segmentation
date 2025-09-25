//#############################################################

// ### 1. LOAD THE MODEL IMMEDIATELY WHEN THE PAGE LOADS

//#############################################################

// Helper functions for UI
function setStatus(text) {
	$('#status-text').text(text);
}

function setProgress(value) {
	const percentage = Math.round(value * 100);
	$('#progress-bar').css('width', percentage + '%').text(percentage + '%');
	$('#progress-container').show(); // ตรวจสอบให้แน่ใจว่า Progress Bar แสดง
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
			'https://muaxzinn.github.io/sblt-cas/Skin_Lesion/final_model_kaggle_version1/model.json',
			{ onProgress: setProgress }
		);
		// เมื่อโหลดเสร็จ ให้ซ่อน Progress Bar และแสดงสถานะพร้อมใช้งาน
		$('#progress-container').hide();
		setStatus('โมเดลพร้อมใช้งาน ✅');
		
		predictOnLoad();
	} catch (e) {
		setStatus('โหลดโมเดลล้มเหลว ❌');
		console.error(e);
	}
})();



	

//######################################################################

// ### 2. MAKE A PREDICTION ON THE FRONT PAGE IMAGE WHEN THE PAGE LOADS

//######################################################################



// The model images have size 96x96

// This code is triggered when the predict button is clicked i.e.
// we simulate a click on the predict button.
$("#predict-button").click(async function () {
	
	let image = undefined;
	
	image = $('#selected-image').get(0);
	
	// Pre-process the image
	// ใช้ tf.browser.fromPixels และปรับค่าสีให้อยู่ในช่วง [-1, 1] ซึ่งเป็นมาตรฐานสำหรับ MobileNet
	let tensor = tf.browser.fromPixels(image)
		.resizeNearestNeighbor([224, 224])
		.toFloat()
		.sub(127.5)
		.div(127.5)
		.expandDims();
	// Pass the tensor to the model and call predict on it.
	// Predict returns a tensor.
	// data() loads the values of the output tensor and returns
	// a promise of a typed array when the computation is complete.
	// Notice the await and async keywords are used together.
	
	// TARGET_CLASSES is defined in the target_clssses.js file.
	// There's no need to load this file because it was imported in index.html
	let predictions = await model.predict(tensor).data();
	let top5 = Array.from(predictions)
		.map(function (p, i) { // this is Array.map
			return {
				probability: p,
				className: TARGET_CLASSES[i] 
			};
				
			
		}).sort(function (a, b) {
			return b.probability - a.probability;
				
		}).slice(0, 3);
	

		// Append the file name to the prediction list
		var file_name = 'ภาพตัวอย่าง.jpg';
		$("#prediction-list").append(`<li class="w3-text-blue fname-font" style="list-style-type:none;">${file_name}</li>`);
		
		//$("#prediction-list").empty();
		top5.forEach(function (p) {
		
			// ist-style-type:none removes the numbers.
			// https://www.w3schools.com/html/html_lists.asp
			$("#prediction-list").append(`<li style="list-style-type:none;">${p.className}: ${p.probability.toFixed(3)}</li>`);
		
			
		});
	
	
});



//######################################################################

// ### 3. READ THE IMAGES THAT THE USER SELECTS

// Then direct the code execution to app_batch_prediction_code.js

//######################################################################




// This listens for a change. It fires when the user submits images.

$("#image-selector").change(async function () {
	
	// the FileReader reads one image at a time
	fileList = $("#image-selector").prop('files');
	
	//$("#prediction-list").empty();
	
	// Start predicting
	// This function is in the app_batch_prediction_code.js file.
	model_processArray(fileList);
	
});
