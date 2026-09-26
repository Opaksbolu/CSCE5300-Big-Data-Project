# CSCE 5300 Parallel K-Means
# Local Spark/PySpark development environment setup.
#
# Usage:
#   source scripts/setup_env.zsh

if [[ "${ZSH_EVAL_CONTEXT:-}" != *:file ]]; then
    echo "ERROR: This script must be sourced:"
    echo "  source scripts/setup_env.zsh"
    return 1 2>/dev/null || exit 1
fi

SCRIPT_DIR="${0:A:h}"
PROJECT_ROOT="${SCRIPT_DIR:h}"

VENV_DIR="${PROJECT_ROOT}/.venv"
SPARK_INSTALLATION="/opt/spark"

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "ERROR: Python virtual environment not found:"
    echo "  ${VENV_DIR}"
    return 1
fi

if [[ ! -d "${SPARK_INSTALLATION}" ]]; then
    echo "ERROR: Spark installation not found:"
    echo "  ${SPARK_INSTALLATION}"
    return 1
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "ERROR: Virtual environment activation script not found."
    return 1
fi

source "${VENV_DIR}/bin/activate"

export SPARK_HOME="${SPARK_INSTALLATION}"

PY4J_ZIP="$(find "${SPARK_HOME}/python/lib" \
    -maxdepth 1 \
    -name 'py4j*.zip' \
    -print \
    -quit)"

if [[ -z "${PY4J_ZIP}" ]]; then
    echo "ERROR: Py4J archive was not found under:"
    echo "  ${SPARK_HOME}/python/lib"
    return 1
fi

export PYTHONPATH="${SPARK_HOME}/python:${PY4J_ZIP}"

echo "CSCE 5300 Spark environment configured."
echo "Project root: ${PROJECT_ROOT}"
echo "Python: $(command -v python)"
echo "SPARK_HOME: ${SPARK_HOME}"
