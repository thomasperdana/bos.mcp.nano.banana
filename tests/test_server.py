import pytest
from fastapi.testclient import TestClient
from server import app, generated_images_store
from unittest.mock import patch

client = TestClient(app)

@pytest.fixture(autouse=True)
def clear_image_store():
    generated_images_store.clear()

def test_upload_file(tmp_path):
    file_content = b"test file content"
    file_path = tmp_path / "test.txt"
    file_path.write_bytes(file_content)

    with open(file_path, "rb") as f:
        response = client.post("/upload", files={"file": ("test.txt", f, "text/plain")})

    assert response.status_code == 200
    assert "file_path" in response.json()
    assert response.json()["file_path"] == "uploads/test.txt"

@patch('server.agent')
def test_generate_prompt_suggestions(mock_agent):
    mock_agent.graph.invoke.return_value = {'messages': [type('obj', (object,), {'content': 'suggestion1\nsuggestion2'})]}
    response = client.post("/generate-prompt-suggestions", data={"prompt": "a test prompt"})
    assert response.status_code == 200
    json_response = response.json()
    assert "suggestions" in json_response
    assert "thread_id" in json_response
    assert json_response["suggestions"] == ["suggestion1", "suggestion2"]

@patch('server.agent')
def test_generate_image(mock_agent):
    mock_agent.graph.invoke.return_value = {"generated_images": [b"image_data"]}
    response = client.post("/generate-image", data={"thread_id": "123", "selected_prompt": "prompt", "size": "1024x1024", "count": 1})
    assert response.status_code == 200
    json_response = response.json()
    assert "image_id" in json_response
    image_id = json_response['image_id']
    assert image_id in generated_images_store
    assert generated_images_store[image_id] == b"image_data"

def test_get_image():
    generated_images_store["test_id"] = b"image_data"
    response = client.get("/get-image/test_id")
    assert response.status_code == 200
    assert response.content == b"image_data"
    assert response.headers["content-type"] == "image/png"

def test_get_image_not_found():
    response = client.get("/get-image/non_existent_id")
    assert response.status_code == 404

@patch('server.upscaler_agent')
def test_upscale_image(mock_upscaler):
    mock_upscaler.run.return_value = {"upscaled_image": b"upscaled_data"}
    generated_images_store["test_id"] = b"image_data"
    response = client.post("/upscale-image", data={"image_id": "test_id"})
    assert response.status_code == 200
    json_response = response.json()
    assert "image_id" in json_response
    upscaled_id = json_response['image_id']
    assert upscaled_id in generated_images_store
    assert generated_images_store[upscaled_id] == b"upscaled_data"

def test_download_image():
    generated_images_store["test_id"] = b"image_data"
    response = client.get("/download-image/test_id")
    assert response.status_code == 200
    assert response.content == b"image_data"
    assert "attachment" in response.headers["content-disposition"]
