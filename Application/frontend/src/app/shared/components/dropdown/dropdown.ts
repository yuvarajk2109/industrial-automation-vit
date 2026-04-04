import { Component, Input, Output, EventEmitter, HostListener, ElementRef, ViewChild, AfterViewInit, OnDestroy, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';


@Component({
    selector: 'app-dropdown',
    standalone: true,
    imports: [CommonModule, FormsModule],
    templateUrl: './dropdown.html',
    styleUrl: './dropdown.css'
})
export class DropdownComponent implements OnInit, AfterViewInit, OnDestroy {

    @Input() label: string = 'Select...';
    @Input() data: any[] = [];
    @Input() displayKey: string = 'label';
    @Input() valueKey: string = 'value';
    @Input() secondaryDisplayKey: string = '';
    @Input() multiselect: boolean = false;
    @Input() allOrNone: boolean = false;
    @Input() search: boolean = false;
    @Input() selected: any = null;
    @Input() hasError: boolean = false;
    @Input() disabled: boolean = false;
    @Input() clearable: boolean = false;
    @Input() icon: string = ''; // CSS class for icon, e.g. "fa-solid fa-filter"
    @Input() variant: 'input' | 'filter' = 'input';
    @Input() size: 'small' | 'medium' = 'medium';
    @Input() lockedKeys: any[] = [];

    @Output() selectionChange = new EventEmitter<any>();

    @ViewChild('panel') panelRef!: ElementRef;

    isOpen = false;
    searchTerm = '';
    openUpwards = false;
    panelTop: number | null = null;
    panelBottom: number | null = null;
    panelLeft: number = 0;
    panelWidth: number = 0;

    constructor(private elementRef: ElementRef) { }

    ngOnInit(): void {
        window.addEventListener('scroll', this.onScroll, true);
    }

    ngAfterViewInit(): void {
        if (this.panelRef) {
            document.body.appendChild(this.panelRef.nativeElement);
        }
    }

    ngOnDestroy(): void {
        window.removeEventListener('scroll', this.onScroll, true);
        if (this.panelRef && this.panelRef.nativeElement && this.panelRef.nativeElement.parentNode) {
            this.panelRef.nativeElement.parentNode.removeChild(this.panelRef.nativeElement);
        }
    }

    private onScroll = (event: Event): void => {
        if (this.isOpen) {
            // Ignore scrolling inside the dropdown panel itself
            if (event.target instanceof Node && this.panelRef?.nativeElement?.contains(event.target)) {
                return;
            }
            this.isOpen = false;
        }
    };

    get filteredData(): any[] {
        if (!this.search || !this.searchTerm.trim()) return this.data;
        const term = this.searchTerm.toLowerCase();
        return this.data.filter(item => {
            const primary = String(item[this.displayKey] || '').toLowerCase();
            const secondary = this.secondaryDisplayKey
                ? String(item[this.secondaryDisplayKey] || '').toLowerCase()
                : '';
            return primary.includes(term) || secondary.includes(term);
        });
    }

    get displayLabel(): string {
        if (this.multiselect) {
            const sel = this.selectedArray;
            if (sel.length === 0) return '';
            if (sel.length === this.data.length && this.data.length > 0) return `All (${sel.length})`;
            return `${sel.length} selected`;
        } else {
            if (!this.selected) return '';
            const primary = this.selected[this.displayKey] || '';
            if (this.secondaryDisplayKey && this.selected[this.secondaryDisplayKey]) {
                return `${primary} - ${this.selected[this.secondaryDisplayKey]}`;
            }
            return primary;
        }
    }

    get hasSelection(): boolean {
        if (this.multiselect) return this.selectedArray.length > 0;
        return this.selected != null;
    }

    get selectedArray(): any[] {
        if (!this.selected) return [];
        return Array.isArray(this.selected) ? this.selected : [];
    }

    toggle(): void {
        if (this.disabled) return;
        this.isOpen = !this.isOpen;
        if (this.isOpen) {
            this.searchTerm = '';
            setTimeout(() => this.calculatePosition(), 0);
        }
    }

    calculatePosition(): void {
        if (!this.elementRef?.nativeElement) return;
        const triggerEl = this.elementRef.nativeElement.querySelector('.trigger-row');
        if (!triggerEl) return;

        const rect = triggerEl.getBoundingClientRect();
        const spaceBelow = window.innerHeight - rect.bottom;
        const spaceAbove = rect.top;

        this.panelWidth = rect.width;
        this.panelLeft = rect.left;

        if (spaceBelow < 240 && spaceAbove > spaceBelow) {
            this.openUpwards = true;
            this.panelBottom = window.innerHeight - rect.top + 4;
            this.panelTop = null;
        } else {
            this.openUpwards = false;
            this.panelTop = rect.bottom + 4;
            this.panelBottom = null;
        }
    }

    @HostListener('window:scroll')
    @HostListener('window:resize')
    onWindowChange(): void {
        if (this.isOpen) {
            this.calculatePosition();
        }
    }

    isSelected(item: any): boolean {
        if (this.multiselect) {
            return this.selectedArray.some(s => s[this.valueKey] === item[this.valueKey]);
        }
        return this.selected && this.selected[this.valueKey] === item[this.valueKey];
    }

    isLocked(item: any): boolean {
        return this.lockedKeys.includes(item[this.valueKey]);
    }

    selectItem(item: any): void {
        this.selectionChange.emit(item);
        this.isOpen = false;
    }

    toggleItem(item: any, event?: Event): void {
        if (this.isLocked(item)) return;
        const arr = [...this.selectedArray];
        const idx = arr.findIndex(s => s[this.valueKey] === item[this.valueKey]);
        if (idx >= 0) arr.splice(idx, 1);
        else arr.push(item);
        this.selectionChange.emit(arr);
    }

    selectAll(event?: Event): void {
        this.selectionChange.emit([...this.data]);
    }

    clearAll(event?: Event): void {
        const locked = this.selectedArray.filter(s => this.lockedKeys.includes(s[this.valueKey]));
        this.selectionChange.emit(locked);
    }

    clear(event?: Event): void {
        if (event) { event.preventDefault(); }
        this.selectionChange.emit(null);
        this.isOpen = false;
    }

    @HostListener('document:mousedown', ['$event'])
    onDocumentClick(event: MouseEvent): void {
        if (!this.elementRef.nativeElement.contains(event.target)) {
            // Also check if click is inside the detached panel
            if (this.panelRef && this.panelRef.nativeElement.contains(event.target)) {
                return;
            }
            this.isOpen = false;
        }
    }
}
