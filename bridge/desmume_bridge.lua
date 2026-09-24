-- Ponte DeSmuME <-> Python para NSMB DS (ver bridge/nsmb_env.py).
--
-- Protocolo por arquivos, sem rede e sem bibliotecas externas no Lua do DeSmuME:
--   python -> bridge/cmd    : "ping" | "dump <tag> [base] [len]" | "hold <botoes>" | "tap <botoes>" | "stop"
--   lua    -> bridge/ram_<tag>.bin : despejo da RAM do ARM9
--   lua    -> bridge/ack    : eco do comando executado (sinal de termino)
--   lua    -> bridge/trace.csv     : frame + botoes, para correlacionar tempo
--
-- Enderecos no espaco do ARM9; RAM principal comeca em 0x02000000.
-- Este script nao contem NENHUM endereco do jogo de proposito: onde fica o
-- estado e descoberto por diferenca entre dumps (bridge/ramdiff.py).

-- Caminho ABSOLUTO de proposito: quando o script e colado na janela de Lua do
-- DeSmuME (e nao carregado de um arquivo), debug.getinfo devolve um "source"
-- sem diretorio e o protocolo de arquivos cairia no CWD do emulador.
local OUT   = "C:/Users/pedro/Downloads/mario-ds/bridge/"
local BASE  = 0x02000000
local SIZE  = 0x400000
local CHUNK = 0x10000          -- 64 KiB por leitura, para nao estourar memoria

local frame, held, tapped = 0, {}, {}

local function p(name) return OUT .. name end

local function probe_api()
  -- Descobre em tempo de execucao o que esta versao do DeSmuME oferece,
  -- em vez de assumir nomes de outras versoes.
  local rep = {}
  for _, g in ipairs({ "memory", "input", "emu", "gui" }) do
    local t = _G[g]
    if type(t) == "table" then
      local ks = {}
      for k in pairs(t) do ks[#ks+1] = k end
      table.sort(ks)
      rep[#rep+1] = string.format("%s: %s", g, table.concat(ks, ","))
    else
      rep[#rep+1] = string.format("%s: AUSENTE", g)
    end
  end
  local f = io.open(p("apirep.txt"), "w")
  f:write(table.concat(rep, "\n")); f:close()
  print("[nsmb] " .. table.concat(rep, "\n[nsmb] "))
end
probe_api()

local function read_cmd()
  local f = io.open(p("cmd"), "r")
  if not f then return nil end
  local s = f:read("*a"); f:close(); os.remove(p("cmd"))
  return s and s:match("^%s*(.-)%s*$")
end

local function ack(t) local f = io.open(p("ack"), "w"); f:write(tostring(t)); f:close() end

-- readbyterange devolve tabela indexada em 0 OU em 1 dependendo da build;
-- as duas formas sao aceitas aqui.
local function read_range(addr, len)
  local ok, t = pcall(memory.readbyterange, addr, len)
  if not ok or type(t) ~= "table" then return nil end
  if t[0] ~= nil then
    local out = {}
    for i = 0, len - 1 do out[i+1] = t[i] or 0 end
    return out
  end
  return t
end

local function do_dump(tag, base, len)
  local fname = p("ram_" .. tag .. ".bin")
  local f = io.open(fname, "wb")
  if not f then return "falha ao abrir " .. fname end
  local written = 0
  for off = 0, len - 1, CHUNK do
    -- math.min nao existe no Lua 5.1 (o que o DeSmuME 0.9.13 usa)
    local n = math.min and math.min(CHUNK, len - off) or (function()
      local r = len - off
      if r > CHUNK then r = CHUNK end
      return r
    end)()
    local bytes = read_range(base + off, n)
    if not bytes then f:close(); return "readbyterange falhou em 0x" .. string.format("%X", base + off) end
    local s = {}
    for i = 1, n do s[i] = string.char(bytes[i] or 0) end
    f:write(table.concat(s)); written = written + n
  end
  f:close()
  return string.format("dump %s ok 0x%X+0x%X -> %d bytes", tag, base, len, written)
end

local function setbuttons(spec)
  held, tapped = {}, {}
  for tok in spec:gmatch("[%a_]+") do
    if tok:sub(1, 4) == "tap_" then tapped[tok:sub(5)] = true
    else held[tok] = true end
  end
end

-- Amostragem por frame: "sample <tag> <nframes> <end> <end> ..." grava
-- frame,v1,v2,... em sample_<tag>.csv a cada frame, sem precisar despejar a RAM
-- inteira. Enderecos em hexadecimal, espaco do ARM9.
local sampling = nil

local function start_sample(tag, nframes, hexlist)
  local addrs = {}
  for h in hexlist:gmatch("0x[xX]?%x+") do
    addrs[#addrs+1] = tonumber(h)
  end
  if #addrs == 0 then return "sample: nenhum endereco" end
  local f = io.open(p("sample_" .. tag .. ".csv"), "w")
  if not f then return "sample: falha ao abrir arquivo" end
  local hdr = {}
  for i, a in ipairs(addrs) do hdr[i] = string.format("0x%X", a) end
  f:write("frame," .. table.concat(hdr, ",") .. "\n")
  sampling = { f = f, left = nframes, addrs = addrs, startframe = frame }
  return string.format("sample %s iniciado: %d frames x %d enderecos",
                       tag, nframes, #addrs)
end

local function tick_sample()
  if not sampling then return end
  local s = sampling
  local vals = {}
  for i, a in ipairs(s.addrs) do
    -- pcall por endereco: ler um endereco invalido nao pode derrubar o emulador
    local v = 0
    local ok, t = pcall(memory.readbyterange, a, 4)
    if ok and type(t) == "table" then
      local function byteAt(k)
        local x = t[k]
        if x == nil then x = t[k+1] end
        return x or 0
      end
      v = byteAt(0) + byteAt(1)*256 + byteAt(2)*65536 + byteAt(3)*16777216
      if v >= 2147483648 then v = v - 4294967296 end
    end
    vals[i] = v
  end
  local ok, err = pcall(function()
    s.f:write(tostring(frame))
    for i = 1, #vals do s.f:write("," .. vals[i]) end
    s.f:write("\n")
  end)
  if not ok then
    s.f:close(); sampling = nil
    print("[nsmb] sample abortado: " .. tostring(err))
    return
  end
  s.left = s.left - 1
  if s.left <= 0 then
    s.f:close()
    sampling = nil
    ack("sample done")
  end
end

-- Chamado pelo DeSmuME a cada frame para sobrescrever o teclado do jogador.
function oninputpolled()
  local function push(t)
    for k, v in pairs(t) do
      if v then
        if input.setvalue then pcall(input.setvalue, k, true) end
      end
    end
  end
  push(held); push(tapped)
  if input.scan then pcall(input.scan) end
  tapped = {}
end

local trace = io.open(p("trace.csv"), "a")
if trace then trace:write("# frame,held,buttons_on\n"); trace:flush() end

local function log_frame()
  if not trace then return end
  local i = input.getprevious and input.getprevious() or {}
  local on = {}
  for k, v in pairs(i) do if v then on[#on+1] = k end end
  table.sort(on)
  local h = {}
  for k in pairs(held) do h[#h+1] = k end
  table.sort(h)
  trace:write(string.format("%d,%s,%s\n", frame, table.concat(h, "+"), table.concat(on, "+")))
  if frame % 30 == 0 then trace:flush() end
end

print("[nsmb] ponte ativa; aguardando bridge/cmd")

while true do
  local cmd = read_cmd()
  if cmd then
    local op, rest = cmd:match("^(%a+)%s*(.*)$")
    if op == "ping" then
      ack("pong " .. frame)
    elseif op == "status" then
      -- O contador do NOSSO loop nao e o relogio do jogo. Isto consulta o
      -- emulador: framecount real, se esta emulando (pausa) e frames de lag.
      local fc = emu.framecount and emu.framecount() or -1
      local em = emu.emulating and emu.emulating() and 1 or 0
      local lg = emu.lagcount and emu.lagcount() or -1
      ack(string.format("framecount=%d emulating=%d lagcount=%d loop=%d", fc, em, lg, frame))
    elseif op == "dump" then
      local tag, b, l = rest:match("^(%S+)%s*(0x[xX]?%x*)%s*(0x[xX]?%x*)$")
      ack(do_dump(tag or "x", tonumber(b) or BASE, tonumber(l) or SIZE))
    elseif op == "hold" then
      setbuttons(rest); ack("hold " .. rest)
    elseif op == "sample" then
      local tag, nf, hexs = rest:match("^(%S+)%s+(%d+)%s*(.*)$")
      ack(start_sample(tag or "x", tonumber(nf) or 100, hexs or ""))
    elseif op == "release" then
      setbuttons(""); ack("release")
    elseif op == "stop" then
      ack("stop"); break
    else
      ack("comando desconhecido: " .. cmd)
    end
  end
  if emu and emu.frameadvance then emu.frameadvance() elseif emu and emu.waitframe then emu.waitframe() end
  frame = frame + 1
  tick_sample()
  log_frame()
end
if trace then trace:close() end
