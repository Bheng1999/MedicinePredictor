# admin.py
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.admin.models import LogEntry, ADDITION, CHANGE, DELETION
from .models import (
    User, Upload, MedicineData, ModelTrainingInfo, 
    Prediction
)


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ['username', 'email', 'role', 'is_active', 'get_created_at']
    list_filter = ['role', 'is_active', 'created_at']
    search_fields = ['username', 'email']
    fieldsets = BaseUserAdmin.fieldsets + (
        ('Custom Fields', {'fields': ('role', 'created_at')}),
    )
    readonly_fields = ['created_at']
    
    def get_created_at(self, obj):
        """Display created_at in local timezone"""
        from django.utils import timezone
        local_time = timezone.localtime(obj.created_at)
        return local_time.strftime('%Y-%m-%d %H:%M:%S %Z')
    get_created_at.short_description = 'Created At'
    get_created_at.admin_order_field = 'created_at'


@admin.register(Upload)
class UploadAdmin(admin.ModelAdmin):
    list_display = ['filename', 'user', 'uploaded_at']
    list_filter = ['uploaded_at', 'user']
    search_fields = ['filename', 'user__username']
    readonly_fields = ['uploaded_at']
    date_hierarchy = 'uploaded_at'


@admin.register(MedicineData)
class MedicineDataAdmin(admin.ModelAdmin):
    list_display = ['medicine_name', 'brand_name', 'category', 'usage', 'qty', 'date']
    list_filter = ['category', 'usage', 'date']
    search_fields = ['medicine_name', 'brand_name', 'category', 'usage']
    date_hierarchy = 'date'
    list_per_page = 50


@admin.register(ModelTrainingInfo)
class ModelTrainingInfoAdmin(admin.ModelAdmin):
    list_display = ['model_type', 'user', 'accuracy', 'status', 'trained_at']
    list_filter = ['status', 'model_type', 'trained_at']
    search_fields = ['model_type', 'user__username']
    readonly_fields = ['trained_at']
    date_hierarchy = 'trained_at'
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('user', 'model_type', 'status', 'trained_at')
        }),
        ('Hyperparameters', {
            'fields': (
                'learning_rate', 'max_depth', 'n_estimators',
                'subsample', 'colsample_bytree', 'gamma',
                'min_child_weight', 'reg_alpha', 'reg_lambda'
            ),
            'classes': ['collapse']
        }),
        ('Results', {
            'fields': ('accuracy', 'training_time', 'data_points')
        }),
    )


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = [
        'medicine_name','brand_name', 'predicted_demand', 'confidence',
        'date_from', 'date_to', 'user', 'created_at'
    ]
    list_filter = ['predicted_demand', 'created_at', 'user','brand_name']
    search_fields = ['medicine_name', 'brand_name', 'category']
    readonly_fields = ['created_at']
    date_hierarchy = 'created_at'


# Built-in Audit Trail via LogEntry
@admin.register(LogEntry)
class LogEntryAdmin(admin.ModelAdmin):
    """
    Django's built-in audit trail - tracks all admin actions automatically.
    Shows who did what, when, and on which object.
    """
    list_display = [
        'action_time', 'user', 'content_type', 
        'object_repr', 'action_flag', 'change_message'
    ]
    list_filter = ['action_flag', 'content_type', 'action_time']
    search_fields = ['object_repr', 'change_message', 'user__username']
    readonly_fields = [
        'action_time', 'user', 'content_type', 
        'object_id', 'object_repr', 'action_flag', 'change_message'
    ]
    date_hierarchy = 'action_time'
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False
    
    def has_delete_permission(self, request, obj=None):
        return False
    
    def action_flag(self, obj):
        flags = {
            ADDITION: 'Addition',
            CHANGE: 'Change',
            DELETION: 'Deletion',
        }
        return flags.get(obj.action_flag, obj.action_flag)
    action_flag.short_description = 'Action'