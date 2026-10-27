-- Циферблат «Пепелац» (Кин-дза-дза), Mi Band 9 Pro, 336x480
-- Файл собран build_face.py из main_template.lua — правьте шаблон.
local lvgl = require("lvgl")
local dataman = require("dataman")
local math = require("math")

local IMG = SCRIPT_PATH or "/"
local function img(name) return IMG .. "v4_" .. name end   -- префикс версии: см. TAG в build_face.py

local W, H = lvgl.HOR_RES(), lvgl.VER_RES()
local root = lvgl.Object(nil, {
    w = W, h = H, bg_color = 0, bg_opa = lvgl.OPA(100), border_width = 0, pad_all = 0
})
root:clear_flag(lvgl.FLAG.SCROLLABLE)

local function Img(name, x, y)
    local o = root:Image { src = img(name) }
    local w, h = o:get_img_size()
    o:set { x = x, y = y, w = w, h = h }
    return o
end

---------------------------------------------------------------- погода
-- Коды погоды Xiaomi (шкала CMA: 0 ясно, 1 облачно, 2 пасмурно, 3 ливень, 4 гроза, ...)
local WEATHER = {}
local function map(codes, name) for _, c in ipairs(codes) do WEATHER[c] = name end end
map({ 0 }, "sunny")
map({ 1, 2, 18, 20, 29, 30, 31, 32, 33, 49, 53, 54, 55, 56, 57, 58 }, "cloudy")   -- облачно, пасмурно, туман, пыль, дымка
map({ 6, 13, 14, 15, 16, 17, 26, 27, 28, 34, 302 }, "snow")                     -- снег, мокрый снег
map({ 3, 7, 8, 9, 10, 11, 12, 19, 21, 22, 23, 24, 25, 301 }, "rain")            -- дождь, ливень
map({ 4, 5 }, "storm")                                                          -- гроза

-- тонировка человека под освещение фона: { цвет, непрозрачность 0..255 }
local TINT = {
    unknown = { "#76705e", 5 },
    sunny = { "#ffffff", 0 },
    cloudy = { "#213864", 17 },
    snow = { "#113787", 31 },
    rain = { "#041029", 93 },
    storm = { "#020828", 121 },
    night = { "#00071e", 188 }
}

---------------------------------------------------------------- объекты
local bg = Img("bg_unknown.png", 0, 0)

-- луна (только ночью): 30 картинок фаз, moon_00 — новолуние, moon_15 — полнолуние
local MN = 30
local moon = Img("moon_00.png", 10, 10)
moon:add_flag(lvgl.FLAG.HIDDEN)

local NF, FPS = 46, 24
local function kuName(i) return string.format("ku_%02d.png", i) end
local ku = Img(kuName(NF - 1), 57, 158)

local TX, TY = { 17, 83, 149, 186, 252 }, 86
local th1 = Img("t_0.png", TX[1], TY)
local th2 = Img("t_0.png", TX[2], TY)
Img("t_c.png", TX[3], TY)
local tm1 = Img("t_0.png", TX[4], TY)
local tm2 = Img("t_0.png", TX[5], TY)

local DX, DY = { 27, 43, 59, 69, 85 }, 9
local dd1 = Img("s_0.png", DX[1], DY)
local dd2 = Img("s_0.png", DX[2], DY)
Img("s_dot.png", DX[3], DY)
local dm1 = Img("s_0.png", DX[4], DY)
local dm2 = Img("s_0.png", DX[5], DY)

-- заряд: до трёх цифр + «%», выравнивание по правому краю
local BR, BY, SPD, SDW, SPCT, SS = 308, 9, 1, 14, 28, 2
local bd = { Img("s_0.png", 0, BY), Img("s_0.png", 0, BY), Img("s_0.png", 0, BY) }
local bpct = Img("s_pct.png", 0, BY)

---------------------------------------------------------------- состояние
local hour, is24, isPM = -1, 1, 0
local wcode = nil
local year, month, day = 2026, 1, 1
local moonIdx = -1
local battery = 0
local cur = nil

-- возраст луны по дате: юлианский день от новолуния 06.01.2000 18:14 UT
local SYNODIC = 29.530588853
local function moonIndex(h)
    local y, m = year, month
    if m <= 2 then y = y - 1; m = m + 12 end
    local a = y // 100
    local jd = math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + day + (2 - a + a // 4) - 1524.5
    jd = jd + (h - 3) / 24                      -- местное время (МСК) -> UT, точность фазы ~ часы
    local age = (jd - 2451550.26) % SYNODIC
    return math.floor(age / SYNODIC * MN) % MN
end

local function update()
    local h = hour
    if is24 == 0 and h >= 0 then                 -- 12-часовой формат -> 0..23
        h = h % 12
        if isPM ~= 0 then h = h + 12 end
    end
    local name
    if h >= 21 or (h >= 0 and h < 6) then
        name = "night"                          -- 21:00–06:00 — ночь при любой погоде
        local i = moonIndex(h)
        if i ~= moonIdx then
            moonIdx = i
            moon:set { src = img(string.format("moon_%02d.png", i)) }
        end
    else
        name = WEATHER[wcode] or "unknown"
    end
    if name ~= cur then
        cur = name
        bg:set { src = img("bg_" .. name .. ".png") }
        if name == "night" then moon:clear_flag(lvgl.FLAG.HIDDEN) else moon:add_flag(lvgl.FLAG.HIDDEN) end
        local t = TINT[name]
        ku:set { img_recolor = t[1], img_recolor_opa = t[2] }
    end
end

local function showNumber(s, pct)
    local n = #s
    local x = BR - (n * (SDW + SS) + (pct and SPCT or -SS))
    for i = 1, 3 do
        local o = bd[i]
        if i <= n then
            local ch = s:sub(i, i)
            o:set { src = img(ch == "-" and "s_dash.png" or ("s_" .. ch .. ".png")), x = x - SPD }
            o:clear_flag(lvgl.FLAG.HIDDEN)
            x = x + SDW + SS
        else
            o:add_flag(lvgl.FLAG.HIDDEN)
        end
    end
    if pct then
        bpct:set { x = x - SPD }
        bpct:clear_flag(lvgl.FLAG.HIDDEN)
    else
        bpct:add_flag(lvgl.FLAG.HIDDEN)
    end
end

local function showBattery()
    showNumber(tostring(battery), true)
end

---------------------------------------------------------------- данные
local function digit(prefix)
    return function(o, value)
        o:set { src = img(string.format("%s%d.png", prefix, value // 256)) }
    end
end
dataman.subscribe("timeHourHigh", th1, digit("t_"))
dataman.subscribe("timeHourLow", th2, digit("t_"))
dataman.subscribe("timeMinuteHigh", tm1, digit("t_"))
dataman.subscribe("timeMinuteLow", tm2, digit("t_"))
dataman.subscribe("dateDayHigh", dd1, digit("s_"))
dataman.subscribe("dateDayLow", dd2, digit("s_"))
dataman.subscribe("dateMonthHigh", dm1, digit("s_"))
dataman.subscribe("dateMonthLow", dm2, digit("s_"))

dataman.subscribe("systemStatusBattery", bpct, function(o, value)
    battery = value // 256
    if battery > 100 then battery = 100 end
    showBattery()
end)

dataman.subscribe("timeHour", root, function(o, value) hour = value // 256; update() end)
dataman.subscribe("dateYear", moon, function(o, value) year = value // 256; update() end)
dataman.subscribe("dateMonth", moon, function(o, value) month = value // 256; update() end)
dataman.subscribe("dateDay", moon, function(o, value) day = value // 256; update() end)
dataman.subscribe("miscIs24H", root, function(o, value) is24 = value // 256; update() end)
dataman.subscribe("miscIsPM", root, function(o, value) isPM = value // 256; update() end)
dataman.subscribe("weatherCurrentWeather", bg, function(o, value)
    if value >= 256 then value = value // 256 end
    wcode = value
    update()
end)

update()
showBattery()

---------------------------------------------------------------- анимация «КУ»
-- проигрывается один раз при включении экрана и замирает на последнем кадре
local frame = NF - 1
local anim = nil
local function play()
    frame = -1
    local para = {
        run = true,
        start_value = 0,
        end_value = NF - 1,
        duration = NF * 1000 // FPS,
        repeat_count = 0,
        path = "linear",
        exec_cb = function(obj, v)
            if v ~= frame then
                frame = v
                obj:set { src = img(kuName(v)) }
            end
        end,
        done_cb = function(a, obj)
            frame = NF - 1
            obj:set { src = img(kuName(NF - 1)) }
        end
    }
    if anim then
        if pcall(function() anim:start() end) then return end
    end
    anim = ku:Anim(para)
end

play()

function ScreenStateChangedCB(pre, now, reason)
    if pre ~= "ON" and now == "ON" then
        update()
        play()
    end
end
