package com.therealcam.data

/** How a series of frames is merged. All modes are classic, deterministic operations on real frames. */
enum class StackMode(val key: String) {
    NONE("none"),
    MEAN("mean"),
    SIGMA_CLIP("sigma_clip"),
    LUCKY("lucky"),
    LIGHTEN("lighten"),
    BEST_OF("best_of");

    companion object {
        fun of(key: String?): StackMode = entries.firstOrNull { it.key == key } ?: NONE
    }
}

/**
 * Capture profile: chosen by the scene icon of a category. Decides how a series is merged and
 * which constraints apply. Numeric parameters come from the user (manual) or the model (AI).
 */
enum class Profile(
    val key: String,
    val label: String,
    val stackMode: StackMode,
    val align: Boolean,
    val rejectOutlierFrames: Boolean,
    val maxFrames: Int,
    val defaultFrames: Int,
    val focusInfinity: Boolean,
) {
    ASTRO("astro", "Астро", StackMode.SIGMA_CLIP, align = true, rejectOutlierFrames = true, maxFrames = 64, defaultFrames = 12, focusInfinity = true),
    LUNAR("lunar", "Луна", StackMode.LUCKY, align = true, rejectOutlierFrames = true, maxFrames = 64, defaultFrames = 20, focusInfinity = true),
    SKY("sky", "Небо", StackMode.MEAN, align = true, rejectOutlierFrames = true, maxFrames = 16, defaultFrames = 4, focusInfinity = true),
    NIGHT("night", "Ночь", StackMode.SIGMA_CLIP, align = true, rejectOutlierFrames = true, maxFrames = 32, defaultFrames = 8, focusInfinity = false),
    TRAILS("trails", "Следы", StackMode.LIGHTEN, align = false, rejectOutlierFrames = false, maxFrames = 64, defaultFrames = 16, focusInfinity = true),
    LONG("long", "Длинная выдержка", StackMode.MEAN, align = true, rejectOutlierFrames = false, maxFrames = 32, defaultFrames = 8, focusInfinity = false),
    ACTION("action", "Движение", StackMode.BEST_OF, align = false, rejectOutlierFrames = false, maxFrames = 10, defaultFrames = 5, focusInfinity = false),
    LANDSCAPE("landscape", "Пейзаж", StackMode.MEAN, align = true, rejectOutlierFrames = true, maxFrames = 8, defaultFrames = 1, focusInfinity = true),
    MACRO("macro", "Макро", StackMode.BEST_OF, align = false, rejectOutlierFrames = false, maxFrames = 5, defaultFrames = 1, focusInfinity = false),
    PORTRAIT("portrait", "Портрет", StackMode.BEST_OF, align = false, rejectOutlierFrames = false, maxFrames = 5, defaultFrames = 1, focusInfinity = false),
    INDOOR("indoor", "Помещение", StackMode.SIGMA_CLIP, align = true, rejectOutlierFrames = true, maxFrames = 8, defaultFrames = 1, focusInfinity = false),
    DOCUMENT("document", "Документ", StackMode.BEST_OF, align = false, rejectOutlierFrames = false, maxFrames = 3, defaultFrames = 1, focusInfinity = false),
    DEFAULT("default", "Обычный", StackMode.NONE, align = false, rejectOutlierFrames = false, maxFrames = 8, defaultFrames = 1, focusInfinity = false);

    companion object {
        fun of(key: String?): Profile = entries.firstOrNull { it.key == key } ?: DEFAULT
    }
}

/** Built-in scene icon keys (docs/MODEL_FORMAT.md §5). The visual is resolved in ui/common/SceneIcons.kt. */
object SceneKeys {
    data class Entry(val key: String, val label: String, val profile: Profile)

    val all: List<Entry> = listOf(
        Entry("stars", "Звёзды", Profile.ASTRO),
        Entry("milky_way", "Млечный путь", Profile.ASTRO),
        Entry("night_sky", "Ночное небо", Profile.ASTRO),
        Entry("aurora", "Сияние", Profile.ASTRO),
        Entry("moon", "Луна", Profile.LUNAR),
        Entry("clouds", "Облака", Profile.SKY),
        Entry("sky", "Небо", Profile.SKY),
        Entry("sunset", "Закат", Profile.SKY),
        Entry("sun", "Солнце", Profile.SKY),
        Entry("city_night", "Ночной город", Profile.NIGHT),
        Entry("fireworks", "Салют", Profile.NIGHT),
        Entry("concert", "Концерт", Profile.NIGHT),
        Entry("fire", "Огонь", Profile.NIGHT),
        Entry("lightning", "Гроза", Profile.NIGHT),
        Entry("light_trails", "Световые следы", Profile.TRAILS),
        Entry("star_trails", "Звёздные треки", Profile.TRAILS),
        Entry("waterfall", "Водопад", Profile.LONG),
        Entry("water", "Вода", Profile.LONG),
        Entry("sport", "Спорт", Profile.ACTION),
        Entry("pet", "Питомец", Profile.ACTION),
        Entry("bird", "Птицы", Profile.ACTION),
        Entry("car", "Транспорт", Profile.ACTION),
        Entry("kids", "Дети", Profile.ACTION),
        Entry("landscape", "Пейзаж", Profile.LANDSCAPE),
        Entry("mountains", "Горы", Profile.LANDSCAPE),
        Entry("forest", "Лес", Profile.LANDSCAPE),
        Entry("snow", "Снег", Profile.LANDSCAPE),
        Entry("city", "Город", Profile.LANDSCAPE),
        Entry("architecture", "Архитектура", Profile.LANDSCAPE),
        Entry("macro", "Макро", Profile.MACRO),
        Entry("flower", "Цветы", Profile.MACRO),
        Entry("insect", "Насекомые", Profile.MACRO),
        Entry("portrait", "Портрет", Profile.PORTRAIT),
        Entry("people", "Люди", Profile.PORTRAIT),
        Entry("selfie", "Селфи", Profile.PORTRAIT),
        Entry("interior", "Интерьер", Profile.INDOOR),
        Entry("food", "Еда", Profile.INDOOR),
        Entry("document", "Документ", Profile.DOCUMENT),
        Entry("screen", "Экран", Profile.DOCUMENT),
        Entry("other", "Другое", Profile.DEFAULT),
    )

    private val byKey = all.associateBy { it.key }

    fun entry(key: String?): Entry = byKey[key] ?: byKey.getValue("other")
    fun profileOf(key: String?): Profile = entry(key).profile
}
