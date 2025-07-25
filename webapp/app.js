document.addEventListener('DOMContentLoaded', function () {
  const fileInput = document.getElementById('file-input');
  const submitBtn = document.getElementById('submit-btn');
  const outputDiv = document.getElementById('output');

  // Enable button when file is selected
  fileInput.addEventListener('change', () => {
    submitBtn.disabled = !fileInput.files.length;
    outputDiv.innerHTML = '';
  });

  submitBtn.addEventListener('click', async () => {
    if (!fileInput.files.length) return;
    submitBtn.disabled = true;
    outputDiv.innerHTML = 'Processing…';
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('image', file);

    try {
      // Replace the base URL below with the URL where your API is hosted
      const response = await fetch('https://your-api-endpoint.example.com/predict', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }
      const result = await response.json();
      displayResult(result);
    } catch (err) {
      outputDiv.innerHTML = `<p class="error">Error: ${err.message}</p>`;
    } finally {
      submitBtn.disabled = false;
    }
  });

  function displayResult(result) {
    const { predicted_class, probability, all_probs, disclaimer } = result;
    let html = `<h3>Prediction</h3>`;
    html += `<p><strong>${predicted_class}</strong> (confidence: ${(probability * 100).toFixed(2)}%)</p>`;
    html += '<h4>Class probabilities</h4><ul>';
    for (const [cls, prob] of Object.entries(all_probs)) {
      html += `<li>${cls}: ${(prob * 100).toFixed(2)}%</li>`;
    }
    html += '</ul>';
    html += `<p><em>${disclaimer}</em></p>`;
    outputDiv.innerHTML = html;
  }
});
