# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for observability examples (dashboards, alerts, configs)."""

import json
from pathlib import Path

import pytest
import yaml

# Path to observability examples
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples" / "online_serving"
DASHBOARDS_DIR = EXAMPLES_DIR / "dashboards" / "grafana"
OBSERVABILITY_DIR = EXAMPLES_DIR / "observability"


class TestGrafanaDashboards:
    """Tests for Grafana dashboard JSON files."""

    @pytest.fixture
    def dashboard_files(self):
        """Get all dashboard JSON files."""
        return list(DASHBOARDS_DIR.glob("*.json"))

    def test_dashboards_exist(self, dashboard_files):
        """Verify that dashboard files exist."""
        assert len(dashboard_files) >= 4, (
            f"Expected at least 4 dashboards, found {len(dashboard_files)}"
        )

        expected_dashboards = [
            "vllm_overview.json",
            "resource_utilization.json",
            "performance_statistics.json",
            "query_statistics.json",
        ]

        existing_names = [f.name for f in dashboard_files]
        for expected in expected_dashboards:
            assert expected in existing_names, f"Missing dashboard: {expected}"

    @pytest.mark.parametrize("dashboard_name", [
        "vllm_overview.json",
        "resource_utilization.json",
        "performance_statistics.json",
        "query_statistics.json",
    ])
    def test_dashboard_valid_json(self, dashboard_name):
        """Verify dashboards are valid JSON."""
        dashboard_path = DASHBOARDS_DIR / dashboard_name
        if not dashboard_path.exists():
            pytest.skip(f"Dashboard {dashboard_name} not found")

        with open(dashboard_path) as f:
            data = json.load(f)

        # Basic structure validation
        assert "panels" in data, f"{dashboard_name} missing 'panels'"
        assert "title" in data, f"{dashboard_name} missing 'title'"
        assert "uid" in data, f"{dashboard_name} missing 'uid'"

    @pytest.mark.parametrize("dashboard_name", [
        "vllm_overview.json",
        "resource_utilization.json",
    ])
    def test_dashboard_has_prometheus_datasource(self, dashboard_name):
        """Verify dashboards use Prometheus datasource variable."""
        dashboard_path = DASHBOARDS_DIR / dashboard_name
        if not dashboard_path.exists():
            pytest.skip(f"Dashboard {dashboard_name} not found")

        with open(dashboard_path) as f:
            data = json.load(f)

        # Check for datasource variable
        templating = data.get("templating", {})
        variables = templating.get("list", [])

        datasource_var = None
        for var in variables:
            if var.get("name") == "DS_PROMETHEUS" or var.get("type") == "datasource":
                datasource_var = var
                break

        assert datasource_var is not None, (
            f"{dashboard_name} missing Prometheus datasource variable"
        )

    @pytest.mark.parametrize("dashboard_name", [
        "vllm_overview.json",
        "resource_utilization.json",
    ])
    def test_dashboard_panels_have_targets(self, dashboard_name):
        """Verify dashboard panels have query targets."""
        dashboard_path = DASHBOARDS_DIR / dashboard_name
        if not dashboard_path.exists():
            pytest.skip(f"Dashboard {dashboard_name} not found")

        with open(dashboard_path) as f:
            data = json.load(f)

        panels = data.get("panels", [])
        visualization_panels = [
            p for p in panels
            if p.get("type") not in ["row", "text"]
        ]

        for panel in visualization_panels:
            targets = panel.get("targets", [])
            assert len(targets) > 0, (
                f"Panel '{panel.get('title', 'unknown')}' in {dashboard_name} "
                f"has no query targets"
            )


class TestObservabilityConfigs:
    """Tests for observability configuration files."""

    def test_alerts_yaml_valid(self):
        """Verify alerts YAML is valid and well-structured."""
        alerts_path = OBSERVABILITY_DIR / "alerts" / "vllm_alerts.yaml"
        if not alerts_path.exists():
            pytest.skip("Alerts file not found")

        with open(alerts_path) as f:
            data = yaml.safe_load(f)

        # Check structure
        assert "groups" in data, "Alerts missing 'groups'"

        groups = data["groups"]
        assert len(groups) > 0, "No alert groups defined"

        for group in groups:
            assert "name" in group, "Alert group missing 'name'"
            assert "rules" in group, f"Alert group '{group['name']}' missing 'rules'"

            for rule in group["rules"]:
                assert "alert" in rule, "Alert rule missing 'alert' name"
                assert "expr" in rule, f"Alert '{rule['alert']}' missing 'expr'"
                assert "labels" in rule, f"Alert '{rule['alert']}' missing 'labels'"
                assert "severity" in rule["labels"], (
                    f"Alert '{rule['alert']}' missing 'severity' label"
                )

    def test_metrics_yaml_valid(self):
        """Verify organized metrics YAML is valid and well-structured."""
        metrics_path = OBSERVABILITY_DIR / "metrics" / "organized_metrics.yaml"
        if not metrics_path.exists():
            pytest.skip("Metrics file not found")

        with open(metrics_path) as f:
            data = yaml.safe_load(f)

        # Check expected categories
        expected_categories = ["latency", "throughput", "resources", "cache"]
        for category in expected_categories:
            assert category in data, f"Missing metric category: {category}"

        # Validate metric entries
        for category, metrics in data.items():
            if not isinstance(metrics, list):
                continue

            for metric in metrics:
                if isinstance(metric, dict) and "name" in metric:
                    assert "description" in metric, (
                        f"Metric '{metric['name']}' missing description"
                    )

    def test_prometheus_config_valid(self):
        """Verify Prometheus configuration is valid YAML."""
        prometheus_path = OBSERVABILITY_DIR / "prometheus" / "prometheus.yaml"
        if not prometheus_path.exists():
            pytest.skip("Prometheus config not found")

        with open(prometheus_path) as f:
            data = yaml.safe_load(f)

        # Check required sections
        assert "global" in data, "Prometheus config missing 'global'"
        assert "scrape_configs" in data, "Prometheus config missing 'scrape_configs'"

        # Check for vllm job
        scrape_configs = data["scrape_configs"]
        job_names = [config["job_name"] for config in scrape_configs]
        assert "vllm" in job_names, "Prometheus config missing 'vllm' job"

    def test_alertmanager_config_valid(self):
        """Verify Alertmanager configuration is valid YAML."""
        alertmanager_path = OBSERVABILITY_DIR / "alertmanager" / "alertmanager.yaml"
        if not alertmanager_path.exists():
            pytest.skip("Alertmanager config not found")

        with open(alertmanager_path) as f:
            data = yaml.safe_load(f)

        # Check required sections
        assert "route" in data, "Alertmanager config missing 'route'"
        assert "receivers" in data, "Alertmanager config missing 'receivers'"

    def test_docker_compose_valid(self):
        """Verify Docker Compose file is valid YAML."""
        compose_path = OBSERVABILITY_DIR / "docker-compose.yaml"
        if not compose_path.exists():
            pytest.skip("Docker Compose file not found")

        with open(compose_path) as f:
            data = yaml.safe_load(f)

        # Check required sections
        assert "services" in data, "Docker Compose missing 'services'"

        services = data["services"]
        expected_services = ["prometheus", "grafana"]
        for service in expected_services:
            assert service in services, f"Docker Compose missing '{service}' service"


class TestRunbooks:
    """Tests for operational runbooks."""

    @pytest.fixture
    def runbook_files(self):
        """Get all runbook markdown files."""
        runbooks_dir = OBSERVABILITY_DIR / "runbooks"
        if not runbooks_dir.exists():
            return []
        return list(runbooks_dir.glob("*.md"))

    def test_runbooks_exist(self, runbook_files):
        """Verify that runbook files exist."""
        assert len(runbook_files) >= 3, (
            f"Expected at least 3 runbooks, found {len(runbook_files)}"
        )

    @pytest.mark.parametrize("runbook_name", [
        "high-latency.md",
        "memory-pressure.md",
        "high-queue-depth.md",
    ])
    def test_runbook_has_required_sections(self, runbook_name):
        """Verify runbooks have required sections."""
        runbook_path = OBSERVABILITY_DIR / "runbooks" / runbook_name
        if not runbook_path.exists():
            pytest.skip(f"Runbook {runbook_name} not found")

        with open(runbook_path) as f:
            content = f.read()

        # Check for key sections
        required_sections = [
            "## Symptoms",
            "## Investigation Steps",
            "## Remediation",
        ]

        for section in required_sections:
            assert section in content, (
                f"Runbook {runbook_name} missing section: {section}"
            )


class TestDashboardMetricReferences:
    """Tests to verify dashboards reference valid vLLM metrics."""

    # Known vLLM metric prefixes
    VLLM_METRIC_PREFIXES = [
        "vllm:time_to_first_token",
        "vllm:inter_token_latency",
        "vllm:e2e_request_latency",
        "vllm:request_queue_time",
        "vllm:generation_tokens",
        "vllm:prompt_tokens",
        "vllm:kv_cache_usage",
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:num_preemptions",
        "vllm:prefix_cache",
        "vllm:request_prompt_tokens",
        "vllm:request_generation_tokens",
    ]

    @pytest.mark.parametrize("dashboard_name", [
        "vllm_overview.json",
        "resource_utilization.json",
    ])
    def test_dashboard_uses_vllm_metrics(self, dashboard_name):
        """Verify dashboards use vLLM metrics in queries."""
        dashboard_path = DASHBOARDS_DIR / dashboard_name
        if not dashboard_path.exists():
            pytest.skip(f"Dashboard {dashboard_name} not found")

        with open(dashboard_path) as f:
            content = f.read()

        # Check that at least some vLLM metrics are referenced
        found_metrics = []
        for prefix in self.VLLM_METRIC_PREFIXES:
            if prefix in content:
                found_metrics.append(prefix)

        assert len(found_metrics) >= 3, (
            f"Dashboard {dashboard_name} should reference more vLLM metrics. "
            f"Found: {found_metrics}"
        )
