#include <pthread.h> // مكتبة الخيوط
#include <stdio.h>   // مكتبة الإدخال والإخراج القياسية

int main(int argc, char *argv[]) {
    
    int scope; // متغير لتخزين قيمة النطاق (SCS or PCS)
    pthread_attr_t attr; // كائن لتخزين سمات الخيط

    /* 1. تهيئة كائن السمات بالإعدادات الافتراضية للنظام */
    pthread_attr_init(&attr);

    /* 2. استخراج قيمة "نطاق الجدولة" من السمات الافتراضية */
    if (pthread_attr_getscope(&attr, &scope) != 0) {
        // حدث خطأ
        fprintf(stderr, "Unable to get scheduling scope\n");
    } else {
        /* 3. فحص القيمة وطباعة النتيجة */
        if (scope == PTHREAD_SCOPE_PROCESS) {
            printf("Default scope is: PTHREAD_SCOPE_PROCESS (PCS)\n");
            printf("(الخيوط تتنافس داخليًا فقط ضمن هذه العملية)\n");
        } else if (scope == PTHREAD_SCOPE_SYSTEM) {
            printf("Default scope is: PTHREAD_SCOPE_SYSTEM (SCS)\n");
            printf("(الخيوط تتنافس مع كل خيوط النظام على المعالج)\n");
        } else {
            fprintf(stderr, "Illegal scope value.\n");
        }
    }

    /* 4. (مهم) تحرير الموارد التي حجزها كائن السمات */
    pthread_attr_destroy(&attr);

    return 0;
}