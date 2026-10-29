# BridgeClip Local — especificação técnica

**Data:** 2026-10-29

**Status:** aprovada para implementação incremental

**Branch de trabalho:** `feat/local-ai-providers`

**Escopo:** integração local no repositório BridgeClip; não alterar o repositório Creator Studio AI.

## Objetivo

Adicionar um modo de criação por execução que transcreve e seleciona Shorts localmente, sem exigir nem utilizar OpenRouter, Zernio ou APIs pagas. Preservar os modos Quality, Economy e Advanced, aproveitar os scripts locais existentes e o editor/renderizador do BridgeClip, e manter o trabalho incremental.

## Auditoria do estado atual

- O contrato Electron/Python é v3. `bridge/bridge_runner.py` exige `OPENROUTER_API_KEY`; `transcription_service.py` e `intelligence_planner.py` usam OpenRouter.
- `bridge/network_guard.py` bloqueia loopback e endereços privados. Para chamar Ollama sem abrir a rede privada em geral, a única exceção proposta é `127.0.0.1:11434`, apenas no modo Local.
- `Review & edit` ativa Jev. Além da avaliação durante a preparação, a ação “Review again” chama Jev no editor. O editor existente já permite revisar, alterar ranges, marcar candidatos prontos e exportar por FFmpeg.
- Os scripts `scripts/local_candidates.py` e `scripts/local_select.py` estão presentes, não rastreados no worktree, e serão preservados e integrados sem descartá-los. Hoje a seleção retorna IDs e limita a aprovação final a dois cortes; a validação de origem/duração e encerramento narrativo precisará ser endurecida.
- O pipeline já envia progresso por eventos, processa em modo local, grava artefatos na pasta de saída e possui renderizadores locais para áudio, ASS e proporção vertical.
- O venv contém Faster-Whisper 1.2.1, PyAV 18.1.0 e CTranslate2 4.8.4; o par não está fixado no lock do engine. Foi executado `faster_whisper.audio.decode_audio()` contra WAV sintético e o caminho `av.open(..., metadata_errors='ignore')` decodificou 1.600 samples. Isso confirma somente compatibilidade de decodificação nessa combinação, não transcrição real.

## Arquitetura e contratos

### Seleção por job e compatibilidade

Adicionar `local` ao modo de clipping da requisição. O modo é selecionado por execução no Create. Os modos existentes e seus contratos permanecem inalterados.

O Electron passa a emitir contrato v4. O bridge e o engine devem aceitar explicitamente v3 e v4:

- v3 mantém o comportamento atual; modo ausente significa Quality. v3 não pode solicitar Local.
- v4 aceita Quality, Economy, Advanced e Local.
- Projetos existentes do editor mantêm `version: 1`. O campo opcional `provider: "local"` distingue novos projetos locais; projetos antigos sem esse campo preservam Jev e o comportamento atual.

O modo Local aceita somente arquivo selecionado pelo picker nativo e saída vertical 9:16. A validação ocorre no renderer, no processo principal e no bridge. Endpoints/modelos locais não são configuráveis pelo renderer.

### Transcrição

`TranscriptionService.transcribe()` continua retornando o contrato atual `TranscriptionResult`, com `TranscriptSegment` e `TranscriptWord`. O adapter local:

- usa Faster-Whisper Base, CPU e INT8;
- resolve o modelo no cache local e falha com instrução explícita se não estiver instalado; não baixa modelo durante um job;
- mantém o modelo em cache singleton por processo, sem recarregá-lo a cada chunk;
- converte os tempos de palavra para milissegundos inteiros, soma offsets de chunks e valida ordem e limites contra a duração do vídeo;
- não cria custo estimado de API nem speaker labels artificiais.

Fixar e gerar hashes para `faster-whisper==1.2.1`, `av==18.1.0` e `ctranslate2==4.8.4`, mais as dependências transitivas compatíveis com Windows/Python 3.12. O lock deve ser reproduzível. Um teste de regressão decodifica WAV sintético usando o caminho `metadata_errors='ignore'`.

### Planejamento e seleção

`IntelligencePlannerService.plan_clips()` continua retornando `ClipPlanResponse` e `ClipPlanSegment`. O adapter Ollama usa somente `http://127.0.0.1:11434/api/chat` com:

- `model: "qwen3:1.7b"`, `stream: false`, `think: false`;
- schema JSON, timeout limitado e parâmetros determinísticos;
- no máximo uma repetição local para saída malformada; nunca troca para outro provedor.

Os scripts existentes mantêm os argumentos CLI e os objetos JSON de topo `{"candidates": [...]}` e `{"clips": [...]}`. A lógica de geração/seleção será reutilizável pelo engine. Os candidatos são derivados de palavras e timestamps da transcrição; o modelo só pode selecionar IDs conhecidos, nunca criar tempos ou texto de origem.

A saída estruturada de avaliação contém ID, `narrative_closed` boolean, `relevance_score` inteiro entre 0 e 100 e justificativa curta. O score é ordenação editorial, não probabilidade/confiança. A janela de contexto enviada ao Ollama inclui texto posterior ao fim do candidato para avaliar se a explicação continua. Além da avaliação semântica, o limite continua ancorado em palavras/sentenças reais.

O validador Python rejeita schema inválido, ID inexistente/duplicado, encerramento não confirmado, duração fora da configuração, range fora do vídeo, overlap e mais de dois cortes. A faixa padrão é 30–60 segundos; as faixas selecionadas na UI definem os limites usados na geração/validação. São permitidos zero, um ou dois resultados; nenhum corte artificial é acrescentado.

## Fluxo Python → Electron → interface

1. Create envia o modo Local e o caminho retornado pelo picker. A fonte remota fica indisponível nesse modo.
2. O gating do Create depende do modo: Local precisa das dependências/modelos locais, não da chave OpenRouter; os três modos existentes continuam exigindo OpenRouter.
3. `job-start.ts` valida fonte e modo e não aplica o bloqueio por chave para Local. `pipeline-runner.ts` serializa v4 e remove do ambiente do worker segredos OpenRouter e Zernio, mesmo que existam em Settings.
4. `bridge_runner.py` ignora a exigência de chave somente para v4 Local. Erros de Ollama indisponível explicam que é necessário iniciar Ollama e instalar `qwen3:1.7b`, sem incluir paths ou payloads.
5. Os estados/eventos existentes de transcrição e planejamento mostram progresso; não é necessário novo IPC. Logs e mensagens não expõem transcrições, caminhos locais ou credenciais.

No modo Local, desligar Jev/coherence, pesquisa externa, visão paga e upload. Nenhuma publicação automática é adicionada. O app não envia mídia/transcrição para serviços externos.

## Egress e ausência de chamadas externas

`network_guard.py` permite somente socket para o endereço IPv4 literal `127.0.0.1`, porta `11434`, quando `clipping_mode == "local"`. `localhost`, IPv6, outras portas loopback/privadas e destinos públicos permanecem bloqueados. O endpoint Ollama é constante, não fornecido pelo renderer.

Os testes cobrem a allowlist e tentativas bloqueadas. Um teste de integração instrumenta as tentativas de conexão no processo real do bridge e falha diante de qualquer destino diferente do Ollama. Também verifica ausência de OpenRouter, S3 e Zernio, execução sem credenciais, cache Whisper offline e nenhum fallback silencioso. O registro de auditoria contém somente destino IP/porta, nunca conteúdo.

## Review & edit sem Jev

`manual_editor.prepare_project()` grava `provider: "local"` e candidatos selecionados com `review: null` e status `refining`. O usuário pode revisar e ajustar início/fim, enquadramento e legendas e então marcar cada corte como pronto.

`ClipEditor` identifica projeto local, mostra ordem/score editorial/justificativa da seleção, esconde a aba/botão Jev e não exibe aviso de revisão Jev desatualizada. O backend também rejeita `action: "review"` em um projeto local, protegendo a fronteira IPC. O restante do editor/export permanece compartilhado. Projetos antigos continuam mostrando Jev.

## Renderização, saída e resultado vazio

- Automatic usa o caminho de renderização do pipeline existente.
- Review & edit usa `manual_editor.export_clip()` e o mesmo `RenderingService`.
- Ambos produzem 9:16, preservam áudio e aplicam legendas ASS queimadas quando habilitadas; arquivos ficam na saída local configurada.
- Zero candidatos produz resultado local válido com `total_clips: 0` e mensagem amigável, sem projeto editor vazio ou falha de renderização. Um ou dois candidatos usam os fluxos normais.
- Não habilitar upload/publicação automática.

## Arquivos previstos

### Existentes a modificar

- `engine/requirements.txt`, `engine/requirements.lock`
- `engine/clip_engine/config.py`
- `engine/clip_engine/bridge_contract.py`
- `engine/clip_engine/services/transcription_service.py`
- `engine/clip_engine/services/intelligence_planner.py`
- `engine/clip_engine/services/ai_clipping_pipeline.py`
- `engine/clip_engine/services/manual_editor.py`
- `bridge/bridge_runner.py`, `bridge/network_guard.py`
- `scripts/local_candidates.py`, `scripts/local_select.py`
- `src/shared/jobs.ts`, `src/shared/job-contract.ts`, `src/shared/clip-editor.ts`
- `src/main/validation.ts`, `src/main/job-start.ts`, `src/main/pipeline-runner.ts`, `src/main/settings-store.ts`
- `src/renderer/pages/ClipPage.tsx`, `src/renderer/pages/SettingsPage.tsx`
- `src/renderer/components/JobForm.tsx`, `SourcePicker.tsx`, `ClipEditor.tsx`
- stores de setup/draft afetados pelo gating do Create
- docs de arquitetura, transcrição e execução Windows

### Novos

- `engine/clip_engine/services/local_transcription.py`
- `engine/clip_engine/services/local_planning.py`
- testes Python de transcrição, planejamento, rede e integração local
- teste UI/main para seleção Local sem chave
- harness PowerShell de E2E real Windows

Nenhum IPC novo é necessário. Os scripts locais não rastreados só serão staged em seus commits funcionais; nenhum vídeo, transcrição ou credencial será adicionado ao Git.

## Testes e critérios de aceite

### Unitários e integração offline

- Compatibilidade `metadata_errors`, timestamps/offsets e cache de modelo.
- Schema JSON, IDs, limites, duração configurada, source bounds, overlap, encerramento narrativo, resultado zero/um/dois e erro de Ollama.
- Contratos v3 e v4; Local sem chaves; allowlist de rede; ausência de cliente OpenRouter/S3/Zernio.
- Projeto local sem Jev, ajustes de range, render e fluxos existentes preservados.
- Nenhum teste unitário conecta a modelos/API reais.

### E2E real no Windows

Adicionar harness opt-in que recebe um vídeo local escolhido pelo usuário, usa Faster-Whisper e Ollama reais e executa o processamento até o MP4 final. O vídeo nunca é commitado nem enviado; saídas ficam em pasta temporária local.

O E2E valida com `ffprobe`: arquivo não vazio, duração válida, 1080×1920, streams de vídeo e áudio. Extrai frame para verificar legenda ASS renderizada e exige inspeção manual do MP4 para legibilidade e encerramento narrativo. Um audit de sockets no mesmo processo falha para qualquer conexão externa. O E2E real não roda em CI genérico sem modelos locais.

Critério narrativo de regressão: um fixture local/sintético com uma frase pontuada que ainda introduz uma explicação deve ser rejeitado; um pensamento concluído deve ser aceito. A decisão final também será revisada no MP4 do E2E.

### Comandos Windows previstos

```powershell
npm run typecheck
npm run lint
npm run test:bridge
npm run test:renderer
npm run test:main
npm test
npm run build
& '.\engine\.venv\Scripts\python.exe' -m pytest -q engine\tests
.\scripts\test-local-e2e.ps1 -VideoPath 'C:\caminho\local\video.mp4'
```

O smoke real requer Ollama ativo com `qwen3:1.7b`, cache local do modelo Whisper Base e FFmpeg com ASS.

## Ordem incremental

1. **Faster-Whisper:** pins/lock, adapter, cache, timestamps e teste `metadata_errors`; validar decodificação/transcrição local antes do Ollama.
2. **Ollama:** schema e teste de encerramento narrativo, reaproveitamento dos scripts, validador e allowlist.
3. **Contratos/bridge:** v4 com leitura compatível v3, seleção Local, sem chave, fonte local, segredo removido e teste egress.
4. **Pipeline/editor/export:** ambos os workflows, zero candidatos como resultado normal e editor sem Jev.
5. **UI/docs/E2E:** gating, progresso, candidatos, instruções Windows e teste de MP4 real; depois suites completas.

Cada checkpoint recebe um commit pequeno com testes do módulo antes de avançar. A branch permanece `feat/local-ai-providers`; Quality, Economy e Advanced não serão removidos ou reescritos.
