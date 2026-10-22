# Keep the on-device inference data model (accessed via reflection-free code,
# but ProGuard can be over-eager with java.util.zip usage patterns we rely on).
-keep class com.nexus500m.app.model.** { *; }
-dontwarn org.slf4j.**
