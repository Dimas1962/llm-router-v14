"""
Test suite for FastAPI REST API (Phase 5)
"""

import pytest
from fastapi.testclient import TestClient

from api.server import app, initialize_router
from api.models import RouteRequest, FeedbackRequest


# Initialize router before tests
initialize_router()

# Test client
client = TestClient(app)


class TestRootEndpoint:
    """Test root endpoint"""

    def test_root(self):
        """Test root endpoint returns API info"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "name" in data
        assert "version" in data
        assert "docs" in data
        assert data["version"] == "1.4.0"


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check(self):
        """Test health endpoint returns healthy status"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["version"] == "1.4.0"
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0

    def test_health_components(self):
        """Test health endpoint includes component status"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "components" in data
        components = data["components"]

        assert "router" in components
        assert "eagle" in components
        assert "memory" in components
        assert "carrot" in components
        assert "context_manager" in components


class TestRouteEndpoint:
    """Test routing endpoint"""

    def test_route_simple_query(self):
        """Test routing a simple query"""
        request_data = {
            "query": "Write a Python function to print hello world"
        }

        response = client.post("/route", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "model" in data
        assert "confidence" in data
        assert "reasoning" in data
        assert "alternatives" in data
        assert "metadata" in data
        assert "routing_time_ms" in data

        # Check model selection
        assert data["model"] in [
            'glm-4-9b', 'qwen3-next-80b', 'qwen3-coder-30b',
            'deepseek-coder-16b', 'qwen2.5-coder-7b'
        ]

        # Check confidence
        assert 0.0 <= data["confidence"] <= 1.0

        # Check routing time
        assert data["routing_time_ms"] > 0

    def test_route_with_history(self):
        """Test routing with session history"""
        request_data = {
            "query": "continue the implementation",
            "session_history": ["message 1", "message 2", "message 3"]
        }

        response = client.post("/route", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "model" in data
        assert data["model"] is not None

    def test_route_with_budget(self):
        """Test routing with budget constraint (CARROT)"""
        request_data = {
            "query": "Write a complex distributed system",
            "budget": 50.0
        }

        response = client.post("/route", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "model" in data
        # Should respect budget constraint

    def test_route_with_user_id(self):
        """Test routing with user ID"""
        request_data = {
            "query": "Test query",
            "user_id": "user123"
        }

        response = client.post("/route", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "model" in data

    def test_route_validation_error(self):
        """Test routing with invalid request"""
        # Missing required query field
        request_data = {}

        response = client.post("/route", json=request_data)

        assert response.status_code == 422  # Validation error
        data = response.json()

        assert "error" in data
        assert data["error"] == "ValidationError"

    def test_route_empty_query(self):
        """Test routing with empty query"""
        request_data = {
            "query": ""
        }

        response = client.post("/route", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_route_alternatives_provided(self):
        """Test that alternatives are provided"""
        request_data = {
            "query": "Write a function"
        }

        response = client.post("/route", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "alternatives" in data
        assert isinstance(data["alternatives"], list)
        # May have 0-3 alternatives

    def test_route_metadata_complete(self):
        """Test that metadata is complete"""
        request_data = {
            "query": "Test query for metadata"
        }

        response = client.post("/route", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "metadata" in data
        metadata = data["metadata"]

        # Should have task_type and routing_strategy
        assert isinstance(metadata, dict)


class TestFeedbackEndpoint:
    """Test feedback endpoint"""

    def test_submit_feedback(self):
        """Test submitting feedback"""
        request_data = {
            "query": "Write a Python function",
            "selected_model": "qwen2.5-coder-7b",
            "success": True,
            "task_type": "coding",
            "complexity": 0.3,
            "satisfaction": 0.9
        }

        response = client.post("/feedback", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert data["status"] in ["success", "skipped"]
        assert "message" in data
        assert "elo_updated" in data

    def test_submit_negative_feedback(self):
        """Test submitting negative feedback"""
        request_data = {
            "query": "Test query",
            "selected_model": "qwen2.5-coder-7b",
            "success": False,
            "task_type": "coding",
            "complexity": 0.5,
            "satisfaction": 0.3
        }

        response = client.post("/feedback", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "status" in data

    def test_feedback_validation_error(self):
        """Test feedback with invalid data"""
        # Missing required fields
        request_data = {
            "query": "Test"
        }

        response = client.post("/feedback", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_feedback_invalid_complexity(self):
        """Test feedback with out-of-range complexity"""
        request_data = {
            "query": "Test",
            "selected_model": "qwen2.5-coder-7b",
            "success": True,
            "task_type": "coding",
            "complexity": 1.5,  # Out of range
        }

        response = client.post("/feedback", json=request_data)

        assert response.status_code == 422  # Validation error


class TestModelsEndpoint:
    """Test models listing endpoint"""

    def test_list_models(self):
        """Test listing all models"""
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()

        assert "models" in data
        assert "total_count" in data

        models = data["models"]
        assert len(models) == 5  # We have 5 models
        assert data["total_count"] == 5

    def test_models_structure(self):
        """Test model information structure"""
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()

        models = data["models"]
        assert len(models) > 0

        # Check first model structure
        model = models[0]
        assert "id" in model
        assert "name" in model
        assert "size" in model
        assert "context_window" in model
        assert "speed" in model
        assert "quality" in model
        assert "tier" in model
        assert "use_cases" in model
        assert "frequency" in model
        assert "elo" in model

    def test_models_values(self):
        """Test that model values are valid"""
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()

        for model in data["models"]:
            # Check ranges
            assert 0 < model["context_window"]
            assert 0 < model["speed"]
            assert 0.0 <= model["quality"] <= 1.0
            assert 0.0 <= model["frequency"] <= 1.0
            assert 1000 < model["elo"] < 2500


class TestStatsEndpoint:
    """Test statistics endpoint"""

    def test_get_stats(self):
        """Test getting router statistics"""
        response = client.get("/stats")

        assert response.status_code == 200
        data = response.json()

        # Check component flags
        assert "eagle_enabled" in data
        assert "memory_enabled" in data
        assert "carrot_enabled" in data
        assert "context_manager_enabled" in data

        # Check model count
        assert "model_count" in data
        assert data["model_count"] == 5

    def test_stats_eagle_data(self):
        """Test that Eagle stats are present when enabled"""
        response = client.get("/stats")

        assert response.status_code == 200
        data = response.json()

        if data["eagle_enabled"]:
            assert "eagle_stats" in data
            # May or may not have data depending on feedback

    def test_stats_memory_data(self):
        """Test that memory stats are present when enabled"""
        response = client.get("/stats")

        assert response.status_code == 200
        data = response.json()

        if data["memory_enabled"]:
            assert "memory_stats" in data


class TestErrorHandling:
    """Test error handling"""

    def test_404_not_found(self):
        """Test 404 error for non-existent endpoint"""
        response = client.get("/nonexistent")

        assert response.status_code == 404

    def test_invalid_json(self):
        """Test handling of invalid JSON"""
        response = client.post(
            "/route",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422


class TestOpenAPIDocumentation:
    """Test OpenAPI documentation"""

    def test_openapi_schema(self):
        """Test that OpenAPI schema is available"""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()

        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema

    def test_docs_ui(self):
        """Test that Swagger UI is available"""
        response = client.get("/docs")

        assert response.status_code == 200

    def test_redoc_ui(self):
        """Test that ReDoc UI is available"""
        response = client.get("/redoc")

        assert response.status_code == 200


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios"""

    def test_full_routing_workflow(self):
        """Test complete routing workflow"""
        # 1. Health check
        health = client.get("/health")
        assert health.status_code == 200

        # 2. Route a query
        route_req = {
            "query": "Write a Python sorting function",
            "session_history": []
        }
        route_resp = client.post("/route", json=route_req)
        assert route_resp.status_code == 200
        route_data = route_resp.json()

        selected_model = route_data["model"]

        # 3. Submit feedback
        feedback_req = {
            "query": route_req["query"],
            "selected_model": selected_model,
            "success": True,
            "task_type": "coding",
            "complexity": 0.3
        }
        feedback_resp = client.post("/feedback", json=feedback_req)
        assert feedback_resp.status_code == 200

        # 4. Check stats
        stats_resp = client.get("/stats")
        assert stats_resp.status_code == 200

    def test_multiple_routes_same_session(self):
        """Test multiple routing calls for same session"""
        history = []

        for i in range(3):
            request_data = {
                "query": f"Continue implementation step {i}",
                "session_history": history
            }

            response = client.post("/route", json=request_data)
            assert response.status_code == 200

            # Add to history for next iteration
            history.append(f"Query: {request_data['query']}")
            if len(history) > 10:
                history = history[-10:]  # Keep last 10

    def test_budget_constrained_routing(self):
        """Test routing with different budget constraints"""
        query = "Implement a complex system"

        # High budget
        response_high = client.post("/route", json={
            "query": query,
            "budget": 100.0
        })
        assert response_high.status_code == 200

        # Low budget
        response_low = client.post("/route", json={
            "query": query,
            "budget": 5.0
        })
        assert response_low.status_code == 200

        # Models may differ based on budget


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
