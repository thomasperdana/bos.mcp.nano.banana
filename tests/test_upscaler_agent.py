import pytest
from upscaler_agent import UpscalerAgent

@pytest.fixture
def agent(mocker):
    mock_model = mocker.patch('google.generativeai.GenerativeModel')
    return UpscalerAgent(mock_model)

def test_build_graph(agent):
    assert agent.graph is not None
    assert "upscale_image" in agent.graph.nodes

def test_upscale_image(agent, mocker):
    mock_part = mocker.Mock()
    mock_part.inline_data.data = b"upscaled_image_data"
    mock_response = mocker.Mock()
    mock_response.parts = [mock_part]
    agent.model.generate_content.return_value = mock_response

    state = {"image": b"image_data"}
    result = agent.upscale_image(state)
    assert "upscaled_image" in result
    assert result["upscaled_image"] == b"upscaled_image_data"
    agent.model.generate_content.assert_called_once()
