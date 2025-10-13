#!/bin/bash

# ======================================================
# TESTE AUTOMATIZADO - SIMULAÇÃO DE CARGA LONGA
# Valida a estabilidade da memória em 10 épocas e 28k lotes
# ======================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "TESTE DE VALIDAÇÃO - CARGA LONGA (10 ÉPOCAS / 28k LOTES)"
echo "=========================================="
echo ""

# FASE 1: Compilação (essencial)
echo -e "${BLUE}[FASE 1] Compilando projeto...${NC}"
if dotnet build > build.log 2>&1; then
    echo -e "${GREEN}✅ Compilação bem-sucedida${NC}"
else
    echo -e "${RED}❌ Erro de compilação${NC}"
    tail -20 build.log
    exit 1
fi
echo ""

# FASE 2: Teste Dinâmico (Execução Longa)
echo -e "${BLUE}[FASE 2] Teste de execução...${NC}"
echo "Iniciando treinamento simulado. Isso pode levar vários minutos."
echo "Monitorando RAM..."
echo ""

# Define um tempo limite generoso para a simulação (e.g., 30 minutos)
# O processo .NET irá se encerrar sozinho quando terminar
TIMEOUT_SECONDS=1800 

# Inicia processo em background com o novo argumento
timeout $TIMEOUT_SECONDS dotnet run -- --run-validation-test > test_run.log 2>&1 &
PID=$!

# Array para armazenar leituras de RAM
declare -a RAM_READINGS

# Monitora RAM enquanto o processo estiver vivo
MONITOR_INTERVAL_SECONDS=5
MAX_READINGS=$((TIMEOUT_SECONDS / MONITOR_INTERVAL_SECONDS))
i=0

while kill -0 $PID 2>/dev/null; do
    sleep $MONITOR_INTERVAL_SECONDS
    
    RAM=$(ps -p $PID -o rss= 2>/dev/null || echo 0)
    RAM_MB=$((RAM / 1024))
    RAM_READINGS+=($RAM_MB)
    
    i=$((i+1))
    echo -ne "\rMonitorando... Leitura $i: RAM = ${RAM_MB}MB"
done

# Aguarda o processo terminar completamente para obter o código de saída
wait $PID || true

echo -e "\n\n${GREEN}Processo de teste concluído.${NC}"
echo ""

# ... O resto do script (Fase 4 e 5) para análise de resultados permanece o mesmo ...
# Ele irá analisar o array RAM_READINGS que foi preenchido durante a execução.

# ========================================
# FASE 4: Análise de Resultados
# ========================================

echo -e "${BLUE}[FASE 4] Análise de resultados...${NC}"
echo ""

if [ ${#RAM_READINGS[@]} -lt 10 ]; then
    echo -e "${YELLOW}⚠️  Dados insuficientes para análise (${#RAM_READINGS[@]} leituras).${NC}"
    echo "Processo pode ter falhado ou terminado muito rápido. Verifique o log."
    echo ""
    echo "Últimas linhas do log:"
    tail -30 test_run.log
    exit 1
fi

# (O código de cálculo de estatísticas (min, max, avg, range, growth) permanece o mesmo)
# ...

# ========================================
# FASE 5: Veredito
# ========================================

echo -e "${BLUE}[FASE 5] Veredito final...${NC}"
echo ""

# (O código de veredito (SCORE, if/else) permanece o mesmo, mas podemos ajustar os thresholds)
# ...
# Para um teste mais longo, podemos ser um pouco mais tolerantes com a variação.
RAM_RANGE_THRESHOLD=800
GROWTH_THRESHOLD=400
RAM_MAX_THRESHOLD=10000 # 10GB

# Teste 1: Variação de RAM aceitável
if [ $RAM_RANGE -lt $RAM_RANGE_THRESHOLD ]; then
    echo -e "${GREEN}✅ Variação de RAM: ${RAM_RANGE}MB < ${RAM_RANGE_THRESHOLD}MB (Estável)${NC}"
    # ...
else
    echo -e "${RED}❌ Variação de RAM: ${RAM_RANGE}MB > ${RAM_RANGE_THRESHOLD}MB (Instável!)${NC}"
fi

# Teste 2: Crescimento linear baixo
if [ $GROWTH -lt $GROWTH_THRESHOLD ]; then
    echo -e "${GREEN}✅ Crescimento linear: ${GROWTH}MB < ${GROWTH_THRESHOLD}MB (Sem vazamento aparente)${NC}"
    # ...
else
    echo -e "${RED}❌ Crescimento linear: ${GROWTH}MB > ${GROWTH_THRESHOLD}MB (VAZAMENTO PROVÁVEL!)${NC}"
fi

# (Resto do veredito)
# ...

# Cleanup
rm -f build.log test_run.log validation_test_dataset.txt