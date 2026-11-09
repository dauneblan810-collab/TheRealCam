package com.therealcam.ui.settings

import android.graphics.Color as AColor
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.therealcam.BuildConfig
import com.therealcam.dataset.MIN_DATASET_SAMPLES
import com.therealcam.ui.common.CategoriesScreen
import com.therealcam.ui.common.SceneIcons
import com.therealcam.ui.theme.RealColors
import com.therealcam.ui.theme.RealTheme
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class SettingsActivity : ComponentActivity() {
    private val vm: SettingsViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge(SystemBarStyle.dark(AColor.TRANSPARENT), SystemBarStyle.dark(AColor.BLACK))
        super.onCreate(savedInstanceState)
        setContent {
            RealTheme {
                var categories by remember { mutableStateOf(false) }
                if (categories) {
                    BackHandler { categories = false }
                    val list by vm.categories.collectAsState()
                    CategoriesScreen(list, { categories = false }, { n, i -> vm.createCategory(n, i) }, { vm.updateCategory(it) }, { vm.deleteCategory(it) })
                } else {
                    SettingsScreen(vm, onBack = { finish() }, onCategories = { categories = true })
                }
            }
        }
    }
}

/** Shown before dataset collection is enabled: lists everything that is recorded. */
const val CONSENT_TEXT = """В режиме сбора датасета для каждого снимка сохраняется:

• геометка (если разрешена геопозиция);
• положение телефона: наклон, крен, азимут;
• гироскоп и акселерометр за 1,5 с до кадра, во время выдержки и 0,75 с после;
• параметры кадра: выдержка, ISO, диафрагма, фокус, баланс белого, режимы автоматики;
• то, что вы выставили вручную (или выбрал ИИ), число кадров серии, способ сведения;
• освещённость, давление, температура батареи, высота Солнца и Луны, фаза Луны, время;
• маленькое превью сцены и числовые признаки, которые видит модель;
• ваша оценка 1–10, оценка по критериям и категория.

Всё хранится только на телефоне, в папке приложения (Android/data/com.therealcam/files/TheRealCam). У приложения нет доступа к интернету, поэтому данные никуда не отправляются. Наружу они попадают только при экспорте ZIP, который делаете вы.

В обучение идут только снимки, оценённые по всем критериям. Минимум для обучения — 150 снимков."""

@Composable
private fun SettingsScreen(vm: SettingsViewModel, onBack: () -> Unit, onCategories: () -> Unit) {
    val s by vm.settings.collectAsState()
    val models by vm.models.collectAsState()
    val modelError by vm.modelError.collectAsState()
    val datasetCount by vm.datasetCount.collectAsState()
    val pendingCount by vm.pendingCount.collectAsState()
    val status by vm.status.collectAsState()
    val busy by vm.busy.collectAsState()
    var consent by remember { mutableStateOf(false) }

    val importLauncher = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri -> uri?.let(vm::importModel) }
    val exportLauncher = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/zip")) { uri -> uri?.let(vm::exportDataset) }

    Column(Modifier.fillMaxSize().background(Color.Black).statusBarsPadding()) {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(4.dp)) {
            IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Назад", tint = Color.White) }
            Text("Настройки", fontSize = 20.sp, fontWeight = FontWeight.Bold, color = Color.White)
        }
        if (busy) LinearProgressIndicator(Modifier.fillMaxWidth(), color = RealColors.Accent)
        status?.let {
            LaunchedEffect(it) {
                kotlinx.coroutines.delay(5000)
                vm.clearStatus()
            }
            Text(it, color = Color.Black, fontSize = 13.sp, modifier = Modifier.fillMaxWidth().background(RealColors.Accent).padding(12.dp))
        }
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).navigationBarsPadding().padding(bottom = 24.dp)) {
            Section("Датасет")
            SwitchRow("Сбор датасета", "Запоминать параметры, датчики и оценки снимков для обучения своей модели", s.datasetMode) { on ->
                if (on) consent = true else vm.setDatasetMode(false)
            }
            Column(Modifier.padding(horizontal = 16.dp, vertical = 6.dp)) {
                Text(
                    "Полностью оценено: $datasetCount из $MIN_DATASET_SAMPLES" + (if (datasetCount >= MIN_DATASET_SAMPLES) " — можно обучать" else "") +
                        (if (pendingCount > 0) "\nЖдут оценки: $pendingCount" else ""),
                    color = Color.White, fontSize = 13.sp,
                )
                Spacer(Modifier.height(6.dp))
                LinearProgressIndicator(
                    progress = { (datasetCount / MIN_DATASET_SAMPLES.toFloat()).coerceAtMost(1f) },
                    modifier = Modifier.fillMaxWidth(), color = if (datasetCount >= MIN_DATASET_SAMPLES) RealColors.Ok else RealColors.Accent,
                )
                Spacer(Modifier.height(10.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(enabled = !busy && datasetCount > 0, onClick = {
                        val stamp = SimpleDateFormat("yyyyMMdd_HHmm", Locale.US).format(Date())
                        exportLauncher.launch("TheRealCam_dataset_$stamp.zip")
                    }) { Text("Экспорт ZIP") }
                    OutlinedButton(onClick = onCategories) { Text("Категории") }
                }
                Spacer(Modifier.height(6.dp))
                Text("Папка: ${vm.datasetPath}", color = RealColors.TextDim, fontSize = 11.sp)
            }
            ChoiceRow("Размер снимка в датасете", listOf(1024 to "1024", 1600 to "1600", 2048 to "2048", 4096 to "4096"), s.datasetImageMaxSide, vm::setDatasetImageMaxSide)
            SwitchRow("Оценка после снимка", "Показывать панель оценки сразу после съёмки", s.askRating, vm::setAskRating)

            Section("AI-модели")
            Column(Modifier.padding(horizontal = 16.dp)) {
                Text(
                    "Модель обучается на ПК программой TheRealCam Trainer по вашему датасету или берётся у другого пользователя (.trcm).",
                    color = RealColors.TextDim, fontSize = 12.sp,
                )
                modelError?.let { Text(it, color = RealColors.Bad, fontSize = 12.sp, modifier = Modifier.padding(top = 4.dp)) }
                Spacer(Modifier.height(8.dp))
                Button(enabled = !busy, onClick = { importLauncher.launch(arrayOf("*/*")) }) { Text("Загрузить модель (.trcm)") }
            }
            Row(Modifier.fillMaxWidth().clickable { vm.selectModel(null) }.padding(horizontal = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                RadioButton(selected = s.activeModel == null, onClick = { vm.selectModel(null) })
                Text("Без модели", color = Color.White)
            }
            models.forEach { m ->
                Row(
                    Modifier.fillMaxWidth().clickable(enabled = m.error == null) { vm.selectModel(m.file) }.padding(horizontal = 8.dp, vertical = 2.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    RadioButton(selected = s.activeModel == m.file.name, enabled = m.error == null, onClick = { vm.selectModel(m.file) })
                    Column(Modifier.weight(1f)) {
                        Text(m.name, color = Color.White)
                        Text(
                            m.error ?: ("${m.classes.size} категорий" + (m.samples?.let { " · обучена на $it снимках" } ?: "") + " · ${m.sizeBytes / 1024} КБ"),
                            color = if (m.error != null) RealColors.Bad else RealColors.TextDim, fontSize = 11.sp,
                        )
                        if (m.classes.isNotEmpty()) Row(Modifier.padding(top = 2.dp), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                            m.classes.take(12).forEach { c ->
                                Box(Modifier.size(20.dp).clip(CircleShape).background(RealColors.SurfaceHigh), contentAlignment = Alignment.Center) {
                                    Icon(SceneIcons.of(c.icon), c.name, tint = Color.White, modifier = Modifier.size(13.dp))
                                }
                            }
                        }
                    }
                    IconButton(onClick = { vm.deleteModel(m.file) }) { Icon(Icons.Filled.Delete, "Удалить", tint = RealColors.TextDim) }
                }
            }

            Section("Камера")
            SwitchRow("Геометки", "Записывать координаты в снимки и датасет", s.geotag, vm::setGeotag)
            SwitchRow("Сохранять в галерею телефона", "DCIM/TheRealCam — снимки останутся после удаления приложения", s.saveToGallery, vm::setSaveToGallery)
            SwitchRow("Сетка", null, s.grid, vm::setGrid)
            SwitchRow("Уровень горизонта", null, s.level, vm::setLevel)
            SwitchRow("Спуск кнопками громкости", null, s.volumeShutter, vm::setVolumeShutter)
            ChoiceRow("Качество JPEG", listOf(90 to "90", 95 to "95", 100 to "100"), s.jpegQuality, vm::setJpegQuality)

            Section("О приложении")
            Text(
                "TheRealCam ${BuildConfig.VERSION_NAME}\nИИ выбирает только параметры съёмки и отбирает кадры. " +
                    "Пиксели он не рисует: серии сводятся выравниванием, усреднением и отбрасыванием выбросов по реальным кадрам.",
                color = RealColors.TextDim, fontSize = 12.sp, modifier = Modifier.padding(horizontal = 16.dp),
            )
        }
    }

    if (consent) {
        AlertDialog(
            onDismissRequest = { consent = false },
            title = { Text("Сбор датасета") },
            text = { Text(CONSENT_TEXT, fontSize = 13.sp, modifier = Modifier.verticalScroll(rememberScrollState())) },
            confirmButton = { TextButton(onClick = { consent = false; vm.setDatasetMode(true) }) { Text("Включить") } },
            dismissButton = { TextButton(onClick = { consent = false }) { Text("Отмена") } },
        )
    }
}

@Composable
private fun Section(title: String) {
    Text(
        title.uppercase(), color = RealColors.Accent, fontSize = 12.sp, fontWeight = FontWeight.Bold,
        modifier = Modifier.padding(start = 16.dp, top = 20.dp, bottom = 6.dp),
    )
}

@Composable
private fun SwitchRow(title: String, subtitle: String?, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(
        Modifier.fillMaxWidth().clickable { onChange(!checked) }.padding(horizontal = 16.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(title, color = Color.White)
            subtitle?.let { Text(it, color = RealColors.TextDim, fontSize = 12.sp) }
        }
        Switch(checked, onChange, colors = SwitchDefaults.colors(checkedTrackColor = RealColors.Accent, checkedThumbColor = Color.Black))
    }
}

@Composable
private fun ChoiceRow(title: String, options: List<Pair<Int, String>>, value: Int, onPick: (Int) -> Unit) {
    Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(title, color = Color.White, modifier = Modifier.weight(1f))
        options.forEach { (v, label) ->
            val sel = v == value
            Text(
                label, fontSize = 12.sp, color = if (sel) Color.Black else Color.White,
                modifier = Modifier.padding(start = 4.dp).clip(RoundedCornerShape(10.dp))
                    .background(if (sel) RealColors.Accent else RealColors.SurfaceHigh).clickable { onPick(v) }
                    .padding(horizontal = 8.dp, vertical = 5.dp),
            )
        }
    }
}
