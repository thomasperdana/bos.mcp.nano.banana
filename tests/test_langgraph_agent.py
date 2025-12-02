import pytest
from langgraph_agent import FancylogoGeneratorAgent

@pytest.fixture
def agent(mocker):
    mock_model = mocker.patch('google.generativeai.GenerativeModel')
    return FancylogoGeneratorAgent(mock_model)

def test_build_graph(agent):
    assert agent.graph is not None
    assert "generate_prompt_suggestions" in agent.graph.nodes
    assert "generate_image" in agent.graph.nodes
    assert "router" in agent.graph.nodes

def test_router_generate(agent):
    state = {"prompt": "a test prompt"}
    result = agent.router(state)
    assert result == "generate"

def test_router_continue(agent):
    state = {"selected_prompt": "a selected prompt"}
    result = agent.router(state)
    assert result == "continue"

def test_generate_prompt_suggestions(agent, mocker):
    mock_response = mocker.Mock()
    mock_response.text = "suggestion1\nsuggestion2"
    agent.model.generate_content.return_value = mock_response
    state = {"prompt": "a test prompt"}
    result = agent.generate_prompt_suggestions(state)
    assert "optimized_prompts" in result
    assert result["optimized_prompts"] == ["suggestion1", "suggestion2"]
    agent.model.generate_content.assert_called_once()

def test_generate_image(agent, mocker):
    mock_part = mocker.Mock()
    mock_part.inline_data.data = b"image_data"
    mock_response = mocker.Mock()
    mock_response.parts = [mock_part]
    agent.model.generate_content.return_value = mock_response

    state = {"selected_prompt": "a selected prompt", "image_size": "1024x1024", "image_count": 1}
    result = agent.generate_image(state)
    assert "generated_images" in result
    assert result["generated_images"] == [b"image_data"]
    agent.model.generate_content.assert_called_once()

def test_generate_image_with_image(agent, mocker, tmp_path):
    mock_part = mocker.Mock()
    mock_part.inline_data.data = b"image_data"
    mock_response = mocker.Mock()
    mock_response.parts = [mock_part]
    agent.model.generate_content.return_value = mock_response

    image_path = tmp_path / "test.png"
    image_path.write_bytes(b"test_image")

    state = {"selected_prompt": "a selected prompt", "image_size": "1024x1024", "image_count": 1, "image_path": str(image_path)}
    result = agent.generate_image(state)
    assert "generated_images" in result
    assert result["generated_images"] == [b"image_data"]
    agent.model.generate_content.assert_called_once()
