# kotlinx.serialization
-keepattributes *Annotation*, InnerClasses
-dontnote kotlinx.serialization.AnnotationsKt
-keepclassmembers class kotlinx.serialization.json.** { *** Companion; }
-keepclasseswithmembers class kotlinx.serialization.json.** { kotlinx.serialization.KSerializer serializer(...); }
-keep,includedescriptorclasses class com.therealcam.**$$serializer { *; }
-keepclassmembers class com.therealcam.** { *** Companion; }
-keepclasseswithmembers class com.therealcam.** { kotlinx.serialization.KSerializer serializer(...); }
